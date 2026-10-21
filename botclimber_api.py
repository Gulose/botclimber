"""Minimaler Python-Client fuer die Botclimber API."""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MAX_JUMP_STRENGTH = 15.0


class BotclimberError(RuntimeError):
    """Wird bei Transportfehlern oder einer nicht erfolgreichen API-Antwort ausgeloest."""


@dataclass(frozen=True)
class BotclimberClient:
    """Synchroner, dependency-freier Client fuer die dokumentierten Bot-Endpunkte."""

    api_key: str
    base_url: str = "https://217.160.132.56/api/v1"
    timeout: float = 10.0
    api_key_header: str = "X-API-Key"
    verify_ssl: bool = True

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> Any:
        """Fuehrt einen JSON-Request aus und vereinheitlicht API-Fehler fuer den Bot."""
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urlencode(query)}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            url,
            data=body,
            method=method,
            headers={
                self.api_key_header: self.api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        ssl_context = ssl.create_default_context()
        if not self.verify_ssl:
            ssl_context = ssl._create_unverified_context()

        try:
            with urlopen(request, timeout=self.timeout, context=ssl_context) as response:
                raw = response.read()
        except HTTPError as error:
            raw = error.read()
            try:
                details = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                details = raw.decode("utf-8", errors="replace")
            raise BotclimberError(
                f"{method} {path} failed with HTTP {error.code}: {details}"
            ) from error
        except URLError as error:
            raise BotclimberError(f"Could not reach {url}: {error.reason}") from error

        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BotclimberError(f"Invalid JSON response from {method} {path}") from error

    def join(self) -> dict[str, Any]:
        return self._request("POST", "/bot/join")

    def jump(self, strength: float, angle_deg: float) -> None:
        if not 0 <= strength <= MAX_JUMP_STRENGTH:
            raise ValueError(f"strength must be between 0 and {MAX_JUMP_STRENGTH}")
        self._request(
            "POST",
            "/bot/jump",
            {"strength": strength, "direction": {"angle_deg": angle_deg}},
        )

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/bot/state")

    def reset(self) -> dict[str, Any]:
        return self._request("POST", "/bot/reset")

    def leave(self) -> None:
        self._request("POST", "/bot/leave")

    def stages(self) -> list[dict[str, Any]]:
        return self._request("GET", "/system/stage")

    def leaderboard(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            "/system/leaderboard",
            query={"limit": limit, "offset": offset},
        )
