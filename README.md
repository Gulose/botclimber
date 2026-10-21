# botclimber

Minimaler Python-Client fuer die Botclimber API.

```bash
export BOTCLIMBER_API_KEY="dein-api-key"
python example_bot.py
```

Der Client liegt in [botclimber_api.py](botclimber_api.py) und deckt `join`,
`jump`, `state`, `reset`, `leave`, `stages` und `leaderboard` ab. Die
Sprungstrategie kann darauf aufbauend in einem separaten Bot implementiert
werden. Fuer den lokalen Entwicklungsserver kann `BOTCLIMBER_BASE_URL` auf
`https://localhost:8443/api/v1` gesetzt werden. Falls der Server ein
selbstsigniertes Zertifikat verwendet, setzt das Beispiel standardmaessig
`verify_ssl` aus `BOTCLIMBER_VERIFY_SSL=0`; in produktiven Umgebungen sollte
die Zertifikatspruefung aktiviert bleiben (`BOTCLIMBER_VERIFY_SSL=1`).