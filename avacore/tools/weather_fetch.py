from __future__ import annotations

import requests


def _try_geocode(query: str, timeout: int = 30) -> tuple[float, float, str] | None:
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": query, "count": 1, "language": "en", "format": "json"},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    results = data.get("results") or []
    if not results:
        return None

    item = results[0]
    name = str(item.get("name", query))
    country = str(item.get("country_code", "")).strip()
    resolved_name = f"{name}, {country}" if country else name

    return float(item["latitude"]), float(item["longitude"]), resolved_name


def geocode_location(location: str) -> tuple[float, float, str]:
    query = (location or "").strip()
    if not query:
        raise RuntimeError("Empty location")

    candidates = [query]

    if "," in query:
        short_query = query.split(",", 1)[0].strip()
        if short_query and short_query not in candidates:
            candidates.append(short_query)

    last_error: Exception | None = None

    for candidate in candidates:
        try:
            result = _try_geocode(candidate, timeout=30)
            if result:
                return result
        except Exception as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(f"Location lookup failed for: {query} ({last_error})")

    raise RuntimeError(f"Location not found: {query}")


def fetch_weather(location: str) -> dict:
    lat, lon, resolved_name = geocode_location(location)

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "forecast_days": 2,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    current = data.get("current") or {}
    daily = data.get("daily") or {}

    return {
        "location": resolved_name,
        "latitude": lat,
        "longitude": lon,
        "current_temperature": current.get("temperature_2m"),
        "current_weather_code": current.get("weather_code"),
        "dates": daily.get("time") or [],
        "temp_max": daily.get("temperature_2m_max") or [],
        "temp_min": daily.get("temperature_2m_min") or [],
        "weather_codes": daily.get("weather_code") or [],
    }
