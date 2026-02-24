"""Download and parse SEPTA Regional Rail GTFS into in-memory data structures."""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher

import httpx

from config import GTFS_RAIL_URL, REGIONAL_RAIL_STATIONS, STATION_ALIASES

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StopTime:
    trip_id: str
    arrival_time: str      # HH:MM:SS (may exceed 24:00)
    departure_time: str    # HH:MM:SS (may exceed 24:00)
    stop_id: str
    stop_sequence: int


@dataclass
class Trip:
    route_id: str
    service_id: str
    trip_id: str
    trip_headsign: str
    direction_id: str


@dataclass
class Route:
    route_id: str
    route_short_name: str
    route_long_name: str


@dataclass
class ServiceCalendar:
    service_id: str
    days: list[bool]       # [mon, tue, wed, thu, fri, sat, sun]
    start_date: date
    end_date: date


@dataclass
class GtfsData:
    stops: dict[str, str] = field(default_factory=dict)                # stop_id -> stop_name
    stop_name_to_ids: dict[str, list[str]] = field(default_factory=dict)  # normalized name -> [stop_ids]
    routes: dict[str, Route] = field(default_factory=dict)             # route_id -> Route
    trips: dict[str, Trip] = field(default_factory=dict)               # trip_id -> Trip
    stop_times_by_stop: dict[str, list[StopTime]] = field(default_factory=dict)  # stop_id -> sorted StopTimes
    stop_times_by_trip: dict[str, list[StopTime]] = field(default_factory=dict)  # trip_id -> sorted StopTimes
    calendar: dict[str, ServiceCalendar] = field(default_factory=dict) # service_id -> ServiceCalendar
    calendar_dates: dict[tuple[str, date], int] = field(default_factory=dict)  # (service_id, date) -> exception_type
    canonical_to_stop_ids: dict[str, list[str]] = field(default_factory=dict)  # canonical station name -> [stop_ids]


# ---------------------------------------------------------------------------
# GTFS time helpers
# ---------------------------------------------------------------------------

def gtfs_time_to_seconds(time_str: str) -> int:
    """Convert GTFS HH:MM:SS to seconds since midnight.

    GTFS allows hours >= 24 for trips past midnight.
    """
    parts = time_str.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def seconds_to_display(seconds: int) -> str:
    """Convert seconds since midnight to human-readable time like '8:05 AM'.

    Handles times >= 24:00 by wrapping into the next day.
    """
    seconds = seconds % 86400  # wrap past-midnight times
    h = seconds // 3600
    m = (seconds % 3600) // 60
    period = "AM" if h < 12 else "PM"
    display_h = h % 12
    if display_h == 0:
        display_h = 12
    return f"{display_h}:{m:02d} {period}"


# ---------------------------------------------------------------------------
# Service date check
# ---------------------------------------------------------------------------

def is_service_running(gtfs: GtfsData, service_id: str, query_date: date) -> bool:
    """Check if a service_id runs on the given date, accounting for exceptions."""
    # Check calendar_dates exceptions first
    exc = gtfs.calendar_dates.get((service_id, query_date))
    if exc == 1:  # added
        return True
    if exc == 2:  # removed
        return False

    cal = gtfs.calendar.get(service_id)
    if cal is None:
        return False

    if query_date < cal.start_date or query_date > cal.end_date:
        return False

    weekday = query_date.weekday()  # 0=Monday
    return cal.days[weekday]


# ---------------------------------------------------------------------------
# Station name mapping (canonical SEPTA names -> GTFS stop_ids)
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Normalize a station name for comparison."""
    return name.lower().strip().replace(".", "").replace("'", "")


def _build_canonical_mapping(gtfs: GtfsData) -> dict[str, list[str]]:
    """Map canonical SEPTA station names to GTFS stop_ids.

    Strategy:
    1. Exact normalized match
    2. Canonical name is substring of GTFS name or vice versa
    3. Fuzzy match (SequenceMatcher ratio >= 0.65)
    """
    # Build reverse lookup: normalized GTFS name -> [stop_ids]
    gtfs_names: dict[str, list[str]] = {}
    for stop_id, stop_name in gtfs.stops.items():
        norm = _normalize(stop_name)
        gtfs_names.setdefault(norm, []).append(stop_id)

    mapping: dict[str, list[str]] = {}

    for canonical in REGIONAL_RAIL_STATIONS:
        norm_canon = _normalize(canonical)

        # 1. Exact match
        if norm_canon in gtfs_names:
            mapping[canonical] = gtfs_names[norm_canon]
            continue

        # 2. Substring match
        found = False
        for gtfs_norm, stop_ids in gtfs_names.items():
            if norm_canon in gtfs_norm or gtfs_norm in norm_canon:
                mapping[canonical] = stop_ids
                found = True
                break
        if found:
            continue

        # 3. Fuzzy match
        best_ratio = 0.0
        best_ids: list[str] = []
        for gtfs_norm, stop_ids in gtfs_names.items():
            ratio = SequenceMatcher(None, norm_canon, gtfs_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_ids = stop_ids
        if best_ratio >= 0.65:
            mapping[canonical] = best_ids

    # Also map aliases
    for alias, canonical in STATION_ALIASES.items():
        if canonical in mapping and alias not in mapping:
            mapping[alias] = mapping[canonical]

    return mapping


def resolve_gtfs_stop_ids(station_name: str, gtfs: GtfsData) -> list[str]:
    """Resolve a station name (canonical or user input) to GTFS stop_ids."""
    # Direct lookup in canonical mapping
    if station_name in gtfs.canonical_to_stop_ids:
        return gtfs.canonical_to_stop_ids[station_name]

    # Try normalized lookup
    norm = _normalize(station_name)
    for canon, stop_ids in gtfs.canonical_to_stop_ids.items():
        if _normalize(canon) == norm:
            return stop_ids

    # Try GTFS stop_name_to_ids directly
    if norm in gtfs.stop_name_to_ids:
        return gtfs.stop_name_to_ids[norm]

    # Fuzzy against GTFS stop names
    best_ratio = 0.0
    best_ids: list[str] = []
    for gtfs_norm, stop_ids in gtfs.stop_name_to_ids.items():
        ratio = SequenceMatcher(None, norm, gtfs_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_ids = stop_ids
    if best_ratio >= 0.6:
        return best_ids

    return []


# ---------------------------------------------------------------------------
# GTFS parsing
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    """Parse GTFS date format YYYYMMDD."""
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _read_csv(zf: zipfile.ZipFile, filename: str) -> list[dict[str, str]]:
    """Read a CSV file from a zip into a list of dicts."""
    with zf.open(filename) as f:
        text = io.TextIOWrapper(f, encoding="utf-8-sig")
        return list(csv.DictReader(text))


def _parse_gtfs_zip(data: bytes) -> GtfsData:
    """Parse a GTFS zip file into a GtfsData structure."""
    gtfs = GtfsData()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # stops.txt
        for row in _read_csv(zf, "stops.txt"):
            stop_id = row["stop_id"]
            stop_name = row["stop_name"]
            gtfs.stops[stop_id] = stop_name
            norm = _normalize(stop_name)
            gtfs.stop_name_to_ids.setdefault(norm, []).append(stop_id)

        # routes.txt
        for row in _read_csv(zf, "routes.txt"):
            route = Route(
                route_id=row["route_id"],
                route_short_name=row.get("route_short_name", ""),
                route_long_name=row.get("route_long_name", ""),
            )
            gtfs.routes[route.route_id] = route

        # trips.txt
        for row in _read_csv(zf, "trips.txt"):
            trip = Trip(
                route_id=row["route_id"],
                service_id=row["service_id"],
                trip_id=row["trip_id"],
                trip_headsign=row.get("trip_headsign", ""),
                direction_id=row.get("direction_id", ""),
            )
            gtfs.trips[trip.trip_id] = trip

        # stop_times.txt
        for row in _read_csv(zf, "stop_times.txt"):
            st = StopTime(
                trip_id=row["trip_id"],
                arrival_time=row["arrival_time"],
                departure_time=row["departure_time"],
                stop_id=row["stop_id"],
                stop_sequence=int(row["stop_sequence"]),
            )
            gtfs.stop_times_by_stop.setdefault(st.stop_id, []).append(st)
            gtfs.stop_times_by_trip.setdefault(st.trip_id, []).append(st)

        # Sort stop_times by departure time (by stop) and stop_sequence (by trip)
        for stop_id in gtfs.stop_times_by_stop:
            gtfs.stop_times_by_stop[stop_id].sort(
                key=lambda s: gtfs_time_to_seconds(s.departure_time)
            )
        for trip_id in gtfs.stop_times_by_trip:
            gtfs.stop_times_by_trip[trip_id].sort(key=lambda s: s.stop_sequence)

        # calendar.txt
        for row in _read_csv(zf, "calendar.txt"):
            cal = ServiceCalendar(
                service_id=row["service_id"],
                days=[
                    row["monday"] == "1",
                    row["tuesday"] == "1",
                    row["wednesday"] == "1",
                    row["thursday"] == "1",
                    row["friday"] == "1",
                    row["saturday"] == "1",
                    row["sunday"] == "1",
                ],
                start_date=_parse_date(row["start_date"]),
                end_date=_parse_date(row["end_date"]),
            )
            gtfs.calendar[cal.service_id] = cal

        # calendar_dates.txt
        for row in _read_csv(zf, "calendar_dates.txt"):
            key = (row["service_id"], _parse_date(row["date"]))
            gtfs.calendar_dates[key] = int(row["exception_type"])

    # Build canonical station name -> GTFS stop_id mapping
    gtfs.canonical_to_stop_ids = _build_canonical_mapping(gtfs)

    log.info(
        "GTFS loaded: %d stops, %d trips, %d stop_times, %d routes, %d services",
        len(gtfs.stops),
        len(gtfs.trips),
        sum(len(v) for v in gtfs.stop_times_by_stop.values()),
        len(gtfs.routes),
        len(gtfs.calendar),
    )

    return gtfs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def load_gtfs(http_client: httpx.AsyncClient) -> GtfsData:
    """Download and parse SEPTA Regional Rail GTFS data."""
    log.info("Downloading GTFS from %s", GTFS_RAIL_URL)
    resp = await http_client.get(GTFS_RAIL_URL, follow_redirects=True)
    resp.raise_for_status()
    return _parse_gtfs_zip(resp.content)
