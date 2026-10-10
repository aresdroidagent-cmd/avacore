"""Deterministic local-time windows for unsolicited guardian communication only."""
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

DAYS = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
DEFAULT_SCHEDULE = {day: ['19:00' if index < 5 else '18:00', '20:00']
                    for index, day in enumerate(DAYS)}


def validate_schedule(schedule):
    if not isinstance(schedule, dict) or set(schedule) != set(DAYS):
        raise ValueError('guardian schedule requires all seven weekdays')
    result = {}
    for day in DAYS:
        interval = schedule[day]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError('each guardian window requires start and end')
        start, end = (time.fromisoformat(value) for value in interval)
        if start.tzinfo or end.tzinfo or start >= end:
            raise ValueError('guardian windows require increasing local times without UTC offsets')
        result[day] = [start.isoformat(timespec='minutes'), end.isoformat(timespec='minutes')]
    return result


def window(instant, schedule, timezone='Europe/Zurich'):
    if instant.tzinfo is None:
        raise ValueError('aware guardian timestamp required')
    zone = ZoneInfo(timezone)
    local = instant.astimezone(zone)
    active_end = None
    next_start = None
    for offset in range(8):
        date = local.date() + timedelta(days=offset)
        start, end = (datetime.combine(date, time.fromisoformat(value), zone)
                      for value in schedule[DAYS[date.weekday()]])
        if start <= local < end:
            active_end = end
        if start > local and next_start is None:
            next_start = start
    return active_end, next_start


@dataclass(frozen=True)
class GuardianInteractionState:
    timezone: str
    current_window_open: bool
    next_window_start: str
    last_proactive_message_at: str | None
    messages_sent_today: int
    pending_initiative_count: int
    guardian_available: bool | None  # Unknown unless Roger explicitly postpones contact.
