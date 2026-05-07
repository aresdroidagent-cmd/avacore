from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import recurring_ical_events
from icalendar import Calendar


@dataclass
class CalendarEvent:
    title: str
    start: datetime | date
    end: datetime | date | None
    location: str = ""
    description: str = ""
    all_day: bool = False


def fetch_ics_calendar(ics_url: str, timeout: int = 20) -> Calendar:
    if not ics_url:
        raise ValueError("ICS URL is not configured")

    request = Request(
        ics_url,
        headers={
            "User-Agent": "AvaCore/0.8 calendar briefing",
        },
    )

    with urlopen(request, timeout=timeout) as response:
        raw = response.read()

    return Calendar.from_ical(raw)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _event_datetime(value: Any) -> datetime | date:
    raw = value.dt

    if isinstance(raw, datetime):
        return raw

    if isinstance(raw, date):
        return raw

    raise TypeError(f"unsupported event datetime type: {type(raw)}")


def _normalize_dt(value: datetime | date, tz: ZoneInfo) -> datetime | date:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=tz)
        return value.astimezone(tz)

    return value


def get_events_for_day(
    ics_url: str,
    target_day: date | None = None,
    timezone: str = "Europe/Zurich",
) -> list[CalendarEvent]:
    tz = ZoneInfo(timezone)
    day = target_day or datetime.now(tz).date()

    start_dt = datetime.combine(day, time.min, tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)

    calendar = fetch_ics_calendar(ics_url)

    components = recurring_ical_events.of(calendar).between(start_dt, end_dt)

    events: list[CalendarEvent] = []

    for component in components:
        summary = _to_text(component.get("summary")) or "(ohne Titel)"
        location = _to_text(component.get("location"))
        description = _to_text(component.get("description"))

        dtstart = component.get("dtstart")
        dtend = component.get("dtend")

        if dtstart is None:
            continue

        start_value = _normalize_dt(_event_datetime(dtstart), tz)

        if dtend is not None:
            end_value = _normalize_dt(_event_datetime(dtend), tz)
        else:
            end_value = None

        all_day = isinstance(start_value, date) and not isinstance(start_value, datetime)

        events.append(
            CalendarEvent(
                title=summary,
                start=start_value,
                end=end_value,
                location=location,
                description=description,
                all_day=all_day,
            )
        )

    events.sort(
        key=lambda event: (
            datetime.combine(event.start, time.min, tzinfo=tz)
            if isinstance(event.start, date) and not isinstance(event.start, datetime)
            else event.start
        )
    )

    return events


def format_event_time(event: CalendarEvent, timezone: str = "Europe/Zurich") -> str:
    tz = ZoneInfo(timezone)

    if event.all_day:
        return "ganztägig"

    if isinstance(event.start, datetime):
        start = event.start.astimezone(tz)
        start_text = start.strftime("%H:%M")
    else:
        return "ganztägig"

    if isinstance(event.end, datetime):
        end = event.end.astimezone(tz)
        end_text = end.strftime("%H:%M")
        return f"{start_text}–{end_text}"

    return start_text


def build_daily_calendar_briefing(
    ics_url: str,
    target_day: date | None = None,
    timezone: str = "Europe/Zurich",
) -> dict:
    tz = ZoneInfo(timezone)
    day = target_day or datetime.now(tz).date()

    events = get_events_for_day(
        ics_url=ics_url,
        target_day=day,
        timezone=timezone,
    )

    date_text = day.strftime("%d.%m.%Y")

    if not events:
        return {
            "ok": True,
            "date": day.isoformat(),
            "events": [],
            "briefing": f"Guten Morgen. Für heute, {date_text}, sind keine Kalendertermine eingetragen.",
        }

    lines = [
        f"Guten Morgen. Dein Kalender für heute, {date_text}:",
        "",
    ]

    payload_events = []

    for event in events:
        time_text = format_event_time(event, timezone=timezone)

        line = f"- {time_text}: {event.title}"

        if event.location:
            line += f" — {event.location}"

        lines.append(line)

        payload_events.append(
            {
                "title": event.title,
                "time": time_text,
                "location": event.location,
                "description": event.description,
                "all_day": event.all_day,
            }
        )

    lines.append("")
    lines.append(f"Anzahl Termine: {len(events)}")

    return {
        "ok": True,
        "date": day.isoformat(),
        "events": payload_events,
        "briefing": "\n".join(lines),
    }