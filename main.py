"""Botclimber-Bot mit einfacher Physik und automatischer Sprungauswahl."""

import math
import os
import time
import json
import logging
from collections import deque
from typing import Any

from botclimber_api import BotclimberClient, BotclimberError


MAX_JUMP_STRENGTH = 15.0
GRAVITY = 10.0
MIN_JUMP_STRENGTH = 5.0
STRENGTH_STEP = 0.5
MIN_FLIGHT_CLEARANCE = 1.1
PLAYER_WIDTH = 0.8
PLAYER_HEIGHT = 0.8
START_POSITION_STEP = 0.2
POSITION_LOG_INTERVAL = 0.333
POSITION_LOG_LIMIT = 50
POSITION_LOG_PATH = "position_log.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def format_platform_coordinates(rect: dict[str, Any]) -> str:
    """Formatiert die Plattform als Koordinatenbereich."""
    x = rect.get("x", 0.0)
    y = rect.get("y", 0.0)
    width = rect.get("width", 0.0)
    height = rect.get("height", 0.0)
    return (
        f"(x:{x:.3f}..{x + width:.3f}, "
        f"y:{y:.3f}..{y + height:.3f})"
    )


def platform_top(rect: dict[str, Any]) -> float:
    """Gibt die obere Kante der Plattform zur Landeberechnung zurueck."""
    return rect.get("y", 0.0) + rect.get("height", 0.0)


def calculate_flight_trajectory(
    start: dict[str, float],
    target: dict[str, float],
    strength: float = MAX_JUMP_STRENGTH,
    gravity: float = GRAVITY,
    clearance: float = MIN_FLIGHT_CLEARANCE,
) -> dict[str, float] | None:
    """Berechnet einen Flug mit der geforderten Hoehe ueber dem Ziel."""
    best: dict[str, float] | None = None
    for angle_deg in range(-89, 90):
        angle = math.radians(angle_deg)
        horizontal_speed = strength * math.sin(angle)
        vertical_speed = strength * math.cos(angle)
        height_delta = target["y"] - start["y"]
        discriminant = vertical_speed**2 - 2 * gravity * height_delta
        if discriminant < 0:
            continue

        flight_time = (vertical_speed + math.sqrt(discriminant)) / gravity
        maximum_height = start["y"] + vertical_speed**2 / (2 * gravity)
        if maximum_height < target["y"] + clearance:
            continue
        landing_x = start["x"] + horizontal_speed * flight_time
        error = abs(landing_x - target["x"])
        if best is None or error < best["landing_error"]:
            best = {
                "angle_deg": float(angle_deg),
                "flight_time": flight_time,
                "landing_x": landing_x,
                "landing_error": error,
            }
    return best


def trajectory_hits_platform(
    start: dict[str, float],
    trajectory: dict[str, float],
    platform: dict[str, Any],
    body_width: float,
    body_height: float,
    gravity: float = GRAVITY,
) -> bool:
    """Prueft, ob der Spieler beim Aufsteigen eine Plattform durchquert."""
    rect = platform["rect"]
    platform_coordinates = format_platform_coordinates(rect)
    # y is the player's bottom-center. On ascent, the player's top reaches
    # the platform bottom before the player's bottom reaches it.
    height_delta = rect["y"] - body_height - start["y"]
    angle = math.radians(trajectory["angle_deg"])
    vertical_speed = trajectory["strength"] * math.cos(angle)
    discriminant = vertical_speed**2 - 2 * gravity * height_delta
    if discriminant < 0:
        logger.debug(
            "Kollision %s: NEIN, Plattformhoehe nicht erreichbar "
            "(Diskriminante %.3f).",
            platform_coordinates,
            discriminant,
        )
        return False

    ascending_time = (vertical_speed - math.sqrt(discriminant)) / gravity
    if not 0 < ascending_time < trajectory["flight_time"]:
        logger.debug(
            "Kollision %s: NEIN, Querung liegt nicht im Aufstieg "
            "(t=%.3f, Flugzeit=%.3f).",
            platform_coordinates,
            ascending_time,
            trajectory["flight_time"],
        )
        return False

    horizontal_speed = trajectory["strength"] * math.sin(angle)
    crossing_x = start["x"] + horizontal_speed * ascending_time
    body_left = crossing_x - body_width / 2
    body_right = crossing_x + body_width / 2
    platform_left = rect["x"]
    platform_right = rect["x"] + rect["width"]
    collision = body_right > platform_left and body_left < platform_right
    logger.debug(
        "Kollision %s: %s, Querung x=%.3f, Spieler=[%.3f, %.3f], "
        "Plattform=[%.3f, %.3f], Spielerhoehe=%.3f.",
        platform_coordinates,
        "JA" if collision else "NEIN",
        crossing_x,
        body_left,
        body_right,
        platform_left,
        platform_right,
        body_height,
    )
    return collision


def has_flight_collision(
    start: dict[str, float],
    trajectory: dict[str, float],
    platforms: list[dict[str, Any]],
    target: dict[str, Any],
    body_width: float,
    body_height: float,
) -> bool:
    """Prueft die Zielplattform und alle hoeheren Plattformen auf Kollision."""
    target_y = target["rect"]["y"]
    for platform in platforms:
        rect = platform.get("rect", {})
        platform_coordinates = format_platform_coordinates(rect)
        if platform is target and target_y <= start["y"]:
            logger.debug(
                "Kollisionspruefung %s: UEBERSPRUNGEN (aktuelle Plattform).",
                platform_coordinates,
            )
            continue
        if platform_top(rect) < platform_top(target["rect"]):
            logger.debug(
                "Kollisionspruefung %s: UEBERSPRUNGEN (unter Ziel).",
                platform_coordinates,
            )
            continue
        if trajectory_hits_platform(
            start, trajectory, platform, body_width, body_height
        ):
            logger.debug(
                "Flugbahn ABGELEHNT: Kollision mit Plattform %s.",
                platform_coordinates,
            )
            return True
        logger.debug(
            "Flugbahn: keine Kollision mit Plattform %s.",
            platform_coordinates,
        )
    logger.debug("Flugbahn AKZEPTIERT: keine blockierende Kollision gefunden.")
    return False


def log_jump_plan(
    state: dict[str, Any],
    jump: dict[str, Any],
    next_platform: dict[str, Any] | None = None,
) -> None:
    """Protokolliert Plattformen, Spielerposition und den ausgewaehlten Sprung."""
    position = state.get("position", {})
    current_rect = current_platform(state)["rect"]
    next_rect = (next_platform or jump["platform"])["rect"]
    trajectory = jump["trajectory"]
    logger.info(
        "Naechster Sprung: aktuelle Plattform=%s, naechste Plattform=%s, "
        "Spielerposition=(x=%.3f, y=%.3f), "
        "Winkel=%.1f Grad, Staerke=%.2f, geplante Landeposition=(x=%.3f, y=%.3f).",
        format_platform_coordinates(current_rect),
        format_platform_coordinates(next_rect),
        position.get("x", 0.0),
        position.get("y", 0.0),
        trajectory["angle_deg"],
        trajectory["strength"],
        trajectory["landing_x"],
        platform_top(next_rect),
    )


def target_points(
    position_x: float,
    rect: dict[str, float],
    body_width: float,
) -> list[float]:
    """Ordnet Mittelpunkt und Kanten nach der Entfernung zur aktuellen Position."""
    left_edge = rect["x"] + body_width / 2
    right_edge = rect["x"] + rect["width"] - body_width / 2
    points = [left_edge, right_edge, rect["x"] + rect["width"] / 2]
    return sorted(set(points), key=lambda point: abs(point - position_x))


def platform_start_points(
    rect: dict[str, float],
    body_width: float,
    preferred_x: float | None = None,
) -> list[float]:
    """Erzeugt begehbare Spielerzentren in festen Abstaenden auf einer Plattform."""
    left = rect["x"] + body_width / 2
    right = rect["x"] + rect["width"] - body_width / 2
    if right < left:
        return [rect["x"] + rect["width"] / 2]

    points = [left]
    while points[-1] + START_POSITION_STEP < right:
        points.append(points[-1] + START_POSITION_STEP)
    points.append(right)
    if preferred_x is not None and left <= preferred_x <= right:
        points.append(preferred_x)
    return sorted(set(round(point, 10) for point in points), key=lambda point: (
        abs(point - preferred_x) if preferred_x is not None else 0.0,
        point,
    ))


def find_trajectory_to_platform(
    start: dict[str, float],
    target_platform: dict[str, Any],
    platforms: list[dict[str, Any]],
    ignored_platform: dict[str, Any],
    body_width: float,
    body_height: float,
    target_xs: list[float] | None = None,
) -> dict[str, Any] | None:
    """Sucht die erste gueltige Flugbahn von einem konkreten Standpunkt."""
    rect = target_platform["rect"]
    target_points_to_try = target_xs or target_points(start["x"], rect, body_width)
    for strength_index in range(
        round((MAX_JUMP_STRENGTH - MIN_JUMP_STRENGTH) / STRENGTH_STEP) + 1
    ):
        strength = MIN_JUMP_STRENGTH + strength_index * STRENGTH_STEP
        for target_x in target_points_to_try:
            target = {"x": target_x, "y": platform_top(rect)}
            trajectory = calculate_flight_trajectory(
                start,
                target,
                strength=strength,
                clearance=body_height,
            )
            if trajectory is None or trajectory["landing_error"] > body_width / 2:
                continue
            trajectory["strength"] = strength
            if not has_flight_collision(
                start,
                trajectory,
                platforms,
                ignored_platform,
                body_width,
                body_height,
            ):
                trajectory["target_x"] = target_x
                return trajectory
    return None


def current_platform(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Ermittelt die Plattform unter dem Spieler oder erzeugt den Startboden."""
    position = state.get("position", {})
    dimensions = state.get("dimensions", {})
    body_width = dimensions.get("x", PLAYER_WIDTH)
    body_height = dimensions.get("y", PLAYER_HEIGHT)
    supports = [
        platform
        for platform in state.get("surrounding_platforms", [])
        if platform_top(platform.get("rect", {})) <= position.get("y", 0.0)
        and platform["rect"]["x"] < position.get("x", 0.0) + body_width / 2
        and platform["rect"]["x"] + platform["rect"]["width"]
        > position.get("x", 0.0) - body_width / 2
    ]
    if supports:
        return max(supports, key=lambda platform: platform["rect"]["y"])
    return {
        "id": "current-platform",
        "rect": {
            "x": position.get("x", 0.0) - 5.0,
            "y": position.get("y", 0.0) - body_height,
            "width": 10.0,
            "height": 1.0,
        },
    }


def find_next_jump(state: dict[str, Any]) -> dict[str, Any] | None:
    """Waehlt einen direkten oder ueber eine gleichhohe Plattform fuehrenden Sprung."""
    position = state.get("position", {})
    platforms = state.get("surrounding_platforms", [])
    dimensions = state.get("dimensions", {})
    body_width = dimensions.get("x", PLAYER_WIDTH)
    body_height = dimensions.get("y", PLAYER_HEIGHT)
    current = current_platform(state)
    current_rect = current["rect"]
    current_level = platform_top(current_rect)
    higher_platforms = sorted(
        (
            platform
            for platform in platforms
            if platform is not current
            and platform_top(platform["rect"]) > current_level + 0.5
        ),
        key=lambda platform: platform_top(platform["rect"]),
    )
    if not higher_platforms:
        return None

    next_platform = higher_platforms[0]
    current_position = {
        "x": position.get("x", 0.0),
        "y": position.get("y", current_level),
    }
    trajectory = find_trajectory_to_platform(
        current_position,
        next_platform,
        platforms,
        current,
        body_width,
        body_height,
    )
    if trajectory is not None:
        jump = {"platform": next_platform, "trajectory": trajectory}
        log_jump_plan(state, jump, next_platform)
        return jump

    for start_x in platform_start_points(
        current_rect,
        body_width,
        preferred_x=current_position["x"],
    ):
        start = {"x": start_x, "y": current_level}
        trajectory = find_trajectory_to_platform(
            start,
            next_platform,
            platforms,
            current,
            body_width,
            body_height,
        )
        if trajectory is not None:
            if math.isclose(start_x, current_position["x"], abs_tol=body_width / 2):
                jump = {"platform": next_platform, "trajectory": trajectory}
            else:
                reposition = find_trajectory_to_platform(
                    current_position,
                    current,
                    platforms,
                    current,
                    body_width,
                    body_height,
                    target_xs=[start_x],
                )
                if reposition is None:
                    continue
                jump = {
                    "platform": current,
                    "trajectory": reposition,
                    "planned_platform": next_platform,
                    "planned_trajectory": trajectory,
                }
            log_jump_plan(state, jump, next_platform)
            return jump

    same_level_platforms = [
        platform
        for platform in platforms
        if platform is not current
        and math.isclose(platform_top(platform["rect"]), current_level)
    ]
    for intermediate in same_level_platforms:
        for intermediate_x in platform_start_points(
            intermediate["rect"], body_width
        ):
            onward = find_trajectory_to_platform(
                {"x": intermediate_x, "y": current_level},
                next_platform,
                platforms,
                intermediate,
                body_width,
                body_height,
            )
            if onward is None:
                continue

            first_leg = find_trajectory_to_platform(
                current_position,
                intermediate,
                platforms,
                current,
                body_width,
                body_height,
                target_xs=[intermediate_x],
            )
            if first_leg is None:
                for start_x in platform_start_points(
                    current_rect,
                    body_width,
                    preferred_x=current_position["x"],
                ):
                    first_leg = find_trajectory_to_platform(
                        {"x": start_x, "y": current_level},
                        intermediate,
                        platforms,
                        current,
                        body_width,
                        body_height,
                        target_xs=[intermediate_x],
                    )
                    if first_leg is not None:
                        break
            if first_leg is not None:
                if not math.isclose(
                    first_leg["target_x"],
                    current_position["x"],
                    abs_tol=body_width / 2,
                ):
                    reposition = find_trajectory_to_platform(
                        current_position,
                        current,
                        platforms,
                        current,
                        body_width,
                        body_height,
                        target_xs=[first_leg["target_x"]],
                    )
                    if reposition is None:
                        continue
                    first_leg = reposition
                jump = {
                    "platform": intermediate,
                    "trajectory": first_leg,
                    "planned_platform": next_platform,
                    "planned_trajectory": onward,
                }
                log_jump_plan(state, jump, intermediate)
                return jump
    return None


def write_position_log(
    state: dict[str, Any],
    entries: deque[str],
    path: str = POSITION_LOG_PATH,
) -> None:
    """Schreibt den aktuellen Verlauf und behaelt nur die letzten 50 Eintraege."""
    entries.append(
        json.dumps(
            {
                "timestamp": time.time(),
                "state": state.get("state"),
                "position": state.get("position"),
                "velocity": state.get("velocity"),
                "platforms": [
                    platform.get("rect")
                    for platform in state.get("surrounding_platforms", [])
                ],
            },
            separators=(",", ":"),
        )
    )
    with open(path, "w", encoding="utf-8") as log_file:
        log_file.write("\n".join(entries))
        log_file.write("\n")


def main() -> None:
    api_key = "34566a6a-38fd-4714-8f8c-e86eac7f6abd"
    if not api_key:
        raise SystemExit("Set BOTCLIMBER_API_KEY before running this example")

    client = BotclimberClient(
        api_key=api_key,
        base_url=os.environ.get(
            "BOTCLIMBER_BASE_URL", "https://217.160.132.56/api/v1"
        ),
        verify_ssl=os.environ.get("BOTCLIMBER_VERIFY_SSL", "0") == "1",
    )

    try:
        session = client.join()
        print(f"Joined session {session.get('session_id')}")
        position_log: deque[str] = deque(maxlen=POSITION_LOG_LIMIT)
        next_log_time = 0.0

        while True:
            state = client.state()
            current_time = time.monotonic()
            if current_time >= next_log_time:
                write_position_log(state, position_log)
                next_log_time = current_time + POSITION_LOG_INTERVAL
            if state.get("state", "").upper() == "LANDED":
                next_jump = find_next_jump(state)
                if next_jump is None:
                    print("No reachable higher platform found")
                    time.sleep(0.25)
                    continue

                trajectory = next_jump["trajectory"]
                client.jump(
                    strength=trajectory["strength"],
                    angle_deg=trajectory["angle_deg"],
                )
                target_rect = next_jump["platform"]["rect"]
                print(
                    "Jumping to platform "
                    f"{format_platform_coordinates(target_rect)} "
                    f"at angle {trajectory['angle_deg']:.0f} degrees "
                    f"with strength {trajectory['strength']:.1f}"
                )
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping bot")
    except BotclimberError as error:
        print(f"API error: {error}")
    finally:
        try:
            pass
            # client.leave()
        except BotclimberError:
            pass


if __name__ == "__main__":
    main()
