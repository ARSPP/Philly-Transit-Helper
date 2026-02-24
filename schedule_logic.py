"""Schedule query logic: look up GTFS timetables for future dates."""

from __future__ import annotations

from datetime import date, timedelta

from departure_logic import resolve_station_name
from gtfs_loader import (
    GtfsData,
    gtfs_time_to_seconds,
    is_service_running,
    resolve_gtfs_stop_ids,
    seconds_to_display,
)
from models import ScheduledDeparture, ScheduledTrip


# ---------------------------------------------------------------------------
# Date parsing (no external dependencies)
# ---------------------------------------------------------------------------

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}


def parse_date(text: str) -> date | None:
    """Parse a date string into a date object.

    Supports:
    - "today"
    - "tomorrow"
    - "next monday", "next tuesday", etc.
    - "YYYY-MM-DD"
    - "MM/DD", "MM-DD" (assumes current year)
    - "MM/DD/YYYY", "MM-DD-YYYY"

    Returns None if unparseable.
    """
    s = text.strip().lower()
    today = date.today()

    if s == "today":
        return today
    if s == "tomorrow":
        return today + timedelta(days=1)

    # "next <weekday>"
    if s.startswith("next "):
        day_name = s[5:].strip()
        target = _WEEKDAYS.get(day_name)
        if target is not None:
            days_ahead = (target - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # "next monday" when today is monday = next week
            return today + timedelta(days=days_ahead)

    # "<weekday>" alone (next occurrence)
    target = _WEEKDAYS.get(s)
    if target is not None:
        days_ahead = (target - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return today + timedelta(days=days_ahead)

    # YYYY-MM-DD
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        try:
            return date.fromisoformat(s)
        except ValueError:
            pass

    # MM/DD/YYYY or MM-DD-YYYY
    for sep in ("/", "-"):
        parts = s.split(sep)
        if len(parts) == 3:
            try:
                m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
                if y < 100:
                    y += 2000
                return date(y, m, d)
            except (ValueError, IndexError):
                pass
        # MM/DD or MM-DD (current year)
        if len(parts) == 2:
            try:
                m, d = int(parts[0]), int(parts[1])
                return date(today.year, m, d)
            except (ValueError, IndexError):
                pass

    return None


# ---------------------------------------------------------------------------
# Time parsing for user input
# ---------------------------------------------------------------------------

def parse_time_to_seconds(text: str) -> int | None:
    """Parse a time string like '8:00 AM', '8AM', '14:30' into seconds since midnight."""
    s = text.strip().upper()

    is_pm = "PM" in s
    is_am = "AM" in s
    s = s.replace("PM", "").replace("AM", "").strip()

    # Remove trailing colon if present
    s = s.rstrip(":")

    parts = s.split(":")
    try:
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None

    if is_pm and h != 12:
        h += 12
    elif is_am and h == 12:
        h = 0

    return h * 3600 + m * 60


# ---------------------------------------------------------------------------
# Schedule lookups
# ---------------------------------------------------------------------------

def lookup_schedule(
    gtfs: GtfsData,
    station_name: str,
    query_date: date,
    time_after: int | None = None,    # seconds since midnight
    time_before: int | None = None,   # seconds since midnight
    count: int = 10,
) -> list[ScheduledDeparture]:
    """Look up scheduled departures from a station on a given date.

    Returns up to ``count`` departures sorted by time.
    """
    # Resolve station to GTFS stop_ids
    resolved, _ = resolve_station_name(station_name)
    stop_ids = resolve_gtfs_stop_ids(resolved, gtfs)
    if not stop_ids:
        return []

    results: list[ScheduledDeparture] = []

    for stop_id in stop_ids:
        stop_times = gtfs.stop_times_by_stop.get(stop_id, [])
        for st in stop_times:
            trip = gtfs.trips.get(st.trip_id)
            if not trip:
                continue

            # Check if service runs on this date
            if not is_service_running(gtfs, trip.service_id, query_date):
                continue

            dep_secs = gtfs_time_to_seconds(st.departure_time)

            # Time window filter
            if time_after is not None and dep_secs < time_after:
                continue
            if time_before is not None and dep_secs > time_before:
                continue

            route = gtfs.routes.get(trip.route_id)
            line_name = _route_display_name(route)

            direction = "Inbound" if trip.direction_id == "0" else "Outbound"

            results.append(ScheduledDeparture(
                departure_time=seconds_to_display(dep_secs),
                destination=trip.trip_headsign or "Unknown",
                line=line_name,
                train_number=st.trip_id.split("_")[-1] if "_" in st.trip_id else st.trip_id,
                direction=direction,
            ))

    # Sort by departure time (convert back for sorting)
    results.sort(key=lambda d: _display_to_sort_key(d.departure_time))

    # Deduplicate (same trip appearing from multiple stop_ids for same station)
    seen: set[str] = set()
    unique: list[ScheduledDeparture] = []
    for dep in results:
        key = f"{dep.departure_time}|{dep.destination}|{dep.line}"
        if key not in seen:
            seen.add(key)
            unique.append(dep)

    return unique[:count]


def lookup_trip_schedule(
    gtfs: GtfsData,
    origin_name: str,
    destination_name: str,
    query_date: date,
    time_after: int | None = None,
    count: int = 10,
) -> list[ScheduledTrip]:
    """Look up A-to-B trips on a given date.

    Finds trips that stop at both origin and destination (origin before destination).
    """
    resolved_origin, _ = resolve_station_name(origin_name)
    resolved_dest, _ = resolve_station_name(destination_name)

    origin_stop_ids = set(resolve_gtfs_stop_ids(resolved_origin, gtfs))
    dest_stop_ids = set(resolve_gtfs_stop_ids(resolved_dest, gtfs))

    if not origin_stop_ids or not dest_stop_ids:
        return []

    results: list[ScheduledTrip] = []

    for trip_id, trip in gtfs.trips.items():
        # Check if service runs on this date
        if not is_service_running(gtfs, trip.service_id, query_date):
            continue

        trip_stops = gtfs.stop_times_by_trip.get(trip_id, [])

        # Find origin and destination stops in this trip
        origin_st = None
        dest_st = None
        for st in trip_stops:
            if st.stop_id in origin_stop_ids and origin_st is None:
                origin_st = st
            if st.stop_id in dest_stop_ids and origin_st is not None:
                dest_st = st
                break

        if origin_st is None or dest_st is None:
            continue

        # Origin must come before destination
        if origin_st.stop_sequence >= dest_st.stop_sequence:
            continue

        dep_secs = gtfs_time_to_seconds(origin_st.departure_time)
        arr_secs = gtfs_time_to_seconds(dest_st.arrival_time)

        if time_after is not None and dep_secs < time_after:
            continue

        travel_min = (arr_secs - dep_secs) // 60

        route = gtfs.routes.get(trip.route_id)
        line_name = _route_display_name(route)

        direction = "Inbound" if trip.direction_id == "0" else "Outbound"

        results.append(ScheduledTrip(
            train_number=trip_id.split("_")[-1] if "_" in trip_id else trip_id,
            line=line_name,
            scheduled_departure=seconds_to_display(dep_secs),
            scheduled_arrival=seconds_to_display(arr_secs),
            travel_time_minutes=travel_min if travel_min > 0 else None,
            direction=direction,
        ))

    results.sort(key=lambda t: _display_to_sort_key(t.scheduled_departure))
    return results[:count]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _route_display_name(route) -> str:
    """Get a human-readable line name from a GTFS Route, stripping ' Line' suffix."""
    if route is None:
        return "Unknown"
    name = route.route_long_name or route.route_short_name or "Unknown"
    if name.endswith(" Line"):
        name = name[:-5]
    return name


def _display_to_sort_key(display_time: str) -> int:
    """Convert a display time like '8:05 AM' to a sort key (seconds)."""
    try:
        secs = parse_time_to_seconds(display_time)
        return secs if secs is not None else 0
    except Exception:
        return 0
