"""Business logic: station name resolution, departure ranking, formatting."""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from config import REGIONAL_RAIL_STATIONS, STATION_ALIASES
from models import (
    AlertData,
    ArrivalEntry,
    BusRouteStatus,
    DepartureOption,
    NextToArriveEntry,
    RailLineStatus,
    RouteStatus,
    StationDeparture,
    TrainViewEntry,
    TransitViewBus,
)

# ---------------------------------------------------------------------------
# Station name resolution
# ---------------------------------------------------------------------------

_STATIONS_LOWER: dict[str, str] = {s.lower(): s for s in REGIONAL_RAIL_STATIONS}
_ALIASES_LOWER: dict[str, str] = {k.lower(): v for k, v in STATION_ALIASES.items()}

# Short names for fuzzy matching (strip common suffixes like " Station")
_SUFFIXES_TO_STRIP = (" station", " transportation center", " transit center")
_STATIONS_SHORT: dict[str, str] = {}
for _canon in REGIONAL_RAIL_STATIONS:
    _low = _canon.lower()
    for _suf in _SUFFIXES_TO_STRIP:
        if _low.endswith(_suf):
            _STATIONS_SHORT[_low[: -len(_suf)]] = _canon
            break


def resolve_station_name(user_input: str) -> tuple[str, bool]:
    """Resolve user input to a canonical SEPTA station name.

    Returns ``(canonical_name, was_exact_match)``.

    Strategy (first match wins):
    1. Case-insensitive exact match against known stations
    2. Alias lookup
    3. Substring match (if exactly one station contains the input)
    4. Fuzzy match against both full and short station names
    5. Fallback: return input as-is and let the SEPTA API validate
    """
    text = user_input.strip()
    lower = text.lower()

    # 1. Exact match
    if lower in _STATIONS_LOWER:
        return _STATIONS_LOWER[lower], True

    # 2. Alias
    if lower in _ALIASES_LOWER:
        return _ALIASES_LOWER[lower], True

    # 3. Substring: input is contained in exactly one station name
    matches = [canon for low, canon in _STATIONS_LOWER.items() if lower in low]
    if len(matches) == 1:
        return matches[0], True

    # 4. Fuzzy match — try both full names and short names (without suffixes)
    best_ratio = 0.0
    best_station = ""

    for low, canon in _STATIONS_LOWER.items():
        ratio = SequenceMatcher(None, lower, low).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_station = canon

    for short, canon in _STATIONS_SHORT.items():
        ratio = SequenceMatcher(None, lower, short).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_station = canon

    if best_ratio >= 0.6:
        return best_station, False

    # 5. Pass-through
    return text, False


# ---------------------------------------------------------------------------
# Delay parsing
# ---------------------------------------------------------------------------

_DELAY_RE = re.compile(r"(\d+)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags, decode entities, and collapse whitespace."""
    cleaned = _HTML_TAG_RE.sub(" ", text)
    cleaned = html.unescape(cleaned)
    return " ".join(cleaned.split()).strip()


def parse_delay_text(delay_str: str | None) -> tuple[int | None, str]:
    """Parse a SEPTA delay string into ``(minutes, human_text)``.

    Examples::

        "On time"  -> (0, "On time")
        "3 min"    -> (3, "3 min late")
        "48 min"   -> (48, "48 min late")
        ""         -> (None, "Unknown")
        None       -> (None, "Unknown")
    """
    if not delay_str:
        return None, "Unknown"
    text = delay_str.strip()
    if not text:
        return None, "Unknown"
    if text.lower() in ("on time", "0 min"):
        return 0, "On time"
    m = _DELAY_RE.search(text)
    if m:
        minutes = int(m.group(1))
        return minutes, f"{minutes} min late"
    return None, text


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

# SEPTA uses several time formats across endpoints
_TIME_FORMATS = [
    "%I:%M%p",         # "5:05PM"
    "%I:%M %p",        # "5:05 PM"
    "%Y-%m-%d %H:%M:%S.%f",  # "2026-02-21 16:29:00.000"
]


def _parse_time(time_str: str) -> datetime | None:
    """Try multiple formats to parse a SEPTA time string."""
    text = time_str.strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _travel_minutes(depart: str, arrive: str) -> int | None:
    """Estimate travel time in minutes between departure and arrival."""
    dep = _parse_time(depart)
    arr = _parse_time(arrive)
    if dep is None or arr is None:
        return None
    diff = arr - dep
    if diff.total_seconds() < 0:
        diff += timedelta(days=1)
    return int(diff.total_seconds() / 60)


def _format_time_short(time_str: str) -> str:
    """Convert SEPTA datetime strings to a short human-readable time.

    '2026-02-21 16:29:00.000' -> '4:29 PM'
    '5:05PM' -> '5:05 PM'
    """
    dt = _parse_time(time_str)
    if dt is None:
        return time_str
    return dt.strftime("%-I:%M %p")


# ---------------------------------------------------------------------------
# Departure ranking (for get_departure_options)
# ---------------------------------------------------------------------------


def rank_departure_options(
    entries: list[NextToArriveEntry],
    arrive_by: str | None = None,
    train_view: list[TrainViewEntry] | None = None,
) -> list[DepartureOption]:
    """Rank NextToArrive results into ordered departure options.

    If *arrive_by* is set, each option is tagged with whether its adjusted
    arrival beats the deadline, and a backup count is computed.

    If *train_view* is provided, track numbers are looked up by train number.
    """
    # Parse arrive_by target
    target_dt: datetime | None = None
    if arrive_by:
        target_dt = _parse_time(arrive_by)

    # Build TrainView lookup for track numbers
    tv_lookup: dict[str, TrainViewEntry] = {}
    if train_view:
        for tv in train_view:
            if tv.trainno:
                tv_lookup[tv.trainno] = tv

    options: list[DepartureOption] = []
    for entry in entries:
        delay_min, delay_text = parse_delay_text(entry.orig_delay)
        is_direct = (entry.isdirect or "").lower() in ("true", "yes")
        travel_min = _travel_minutes(
            entry.orig_departure_time, entry.orig_arrival_time
        )

        # Look up track from TrainView
        track: str | None = None
        tv_entry = tv_lookup.get(entry.orig_train)
        if tv_entry:
            track = tv_entry.TRACK or None
            # Use fresher delay from TrainView if available
            if tv_entry.late != (delay_min or 0):
                delay_min = tv_entry.late
                delay_text = "On time" if tv_entry.late == 0 else f"{tv_entry.late} min late"

        # Check arrive_by target
        arrives_by_target: bool | None = None
        if target_dt is not None:
            arr_dt = _parse_time(entry.orig_arrival_time)
            if arr_dt is not None and delay_min is not None:
                adjusted = arr_dt + timedelta(minutes=delay_min)
                arrives_by_target = adjusted <= target_dt

        # Connection info
        connection_info: str | None = None
        if not is_direct and entry.term_line:
            connection_info = (
                f"Connect to {entry.term_line} "
                f"(train {entry.term_train})"
            )

        options.append(
            DepartureOption(
                rank=0,  # set below after sorting
                train_number=entry.orig_train,
                line=entry.orig_line or "Unknown",
                scheduled_departure=_format_time_short(
                    entry.orig_departure_time
                ),
                scheduled_arrival=_format_time_short(
                    entry.orig_arrival_time
                ),
                delay_minutes=delay_min,
                delay_text=delay_text,
                is_direct=is_direct,
                connection_info=connection_info,
                track=track,
                travel_time_minutes=travel_min,
                arrives_by_target=arrives_by_target,
                backup_count=None,  # computed after sorting
                recommendation="",  # set below
            )
        )

    # Sort strategy depends on whether arrive_by is set
    if target_dt is not None:
        # arrive_by mode: viable trains first, then by earliest departure
        options.sort(
            key=lambda o: (
                o.arrives_by_target is not True,  # viable first
                _parse_time(o.scheduled_departure) or datetime.max,
            )
        )
    else:
        # default mode: on-time directs first, then by delay
        options.sort(
            key=lambda o: (
                not o.is_direct,
                o.delay_minutes if o.delay_minutes is not None else 999,
            )
        )

    # Assign ranks
    for i, opt in enumerate(options):
        opt.rank = i + 1

    # Generate recommendations
    if target_dt is not None:
        # arrive_by mode: backup counts + deadline-aware recommendations
        for opt in options:
            if opt.arrives_by_target is True:
                later_on_time = sum(
                    1
                    for o in options
                    if o.rank > opt.rank and o.arrives_by_target is True
                )
                opt.backup_count = later_on_time
                detail = _delay_and_direct_detail(opt)
                if later_on_time == 0:
                    opt.recommendation = f"Last train that arrives on time{detail}"
                else:
                    opt.recommendation = f"Arrives on time — {later_on_time} backup(s) after this{detail}"
            elif opt.arrives_by_target is False:
                opt.recommendation = "Arrives after target time"
    else:
        for i, opt in enumerate(options):
            opt.recommendation = _make_recommendation(opt, i)

    return options


def _make_recommendation(opt: DepartureOption, index: int) -> str:
    """Generate a human-readable recommendation for a departure option."""
    parts: list[str] = []

    if index == 0:
        parts.append("Best option")
    elif index == 1:
        parts.append("Good backup")

    if opt.delay_minutes is not None and opt.delay_minutes == 0:
        parts.append("on time")
    elif opt.delay_minutes is not None and opt.delay_minutes > 0:
        parts.append(f"running {opt.delay_minutes} min late")

    if opt.is_direct:
        parts.append("direct")
    else:
        parts.append("requires connection")

    return ", ".join(parts).capitalize() if parts else "Alternative option"


def _delay_and_direct_detail(opt: DepartureOption) -> str:
    """Build a short suffix like ', direct, on time' for arrive_by recommendations."""
    parts: list[str] = []
    if opt.is_direct:
        parts.append("direct")
    else:
        parts.append("requires connection")
    if opt.delay_minutes is not None and opt.delay_minutes == 0:
        parts.append("on time")
    elif opt.delay_minutes is not None and opt.delay_minutes > 0:
        parts.append(f"running {opt.delay_minutes} min late")
    return (", " + ", ".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# Station departures formatting (for station_departures)
# ---------------------------------------------------------------------------


def format_station_departures(
    arrivals: dict[str, list[ArrivalEntry]],
    train_view: list[TrainViewEntry] | None = None,
) -> list[StationDeparture]:
    """Convert Arrivals API data + optional TrainView enrichment."""

    # Build lookup: train_id -> TrainViewEntry
    tv_lookup: dict[str, TrainViewEntry] = {}
    if train_view:
        for tv in train_view:
            if tv.trainno:
                tv_lookup[tv.trainno] = tv

    departures: list[StationDeparture] = []
    for direction_name, entries in arrivals.items():
        for entry in entries:
            # Skip entries missing critical fields
            if not entry.train_id:
                continue
            delay_min, delay_text = parse_delay_text(entry.status)
            track = entry.track
            platform = entry.platform

            # Enrich from TrainView if available
            if entry.train_id in tv_lookup:
                tv = tv_lookup[entry.train_id]
                # TrainView often has fresher delay data
                if tv.late != (delay_min or 0):
                    delay_min = tv.late
                    delay_text = "On time" if tv.late == 0 else f"{tv.late} min late"
                if tv.TRACK and not track:
                    track = tv.TRACK

            departures.append(
                StationDeparture(
                    direction=direction_name,
                    train_number=entry.train_id,
                    destination=entry.destination or "Unknown",
                    line=entry.line or "Unknown",
                    scheduled_time=_format_time_short(entry.depart_time or entry.sched_time or ""),
                    delay_text=delay_text,
                    delay_minutes=delay_min,
                    track=track,
                    platform=platform,
                    service_type=entry.service_type or "LOCAL",
                )
            )

    return departures


# ---------------------------------------------------------------------------
# Alert / status formatting (for check_status)
# ---------------------------------------------------------------------------


def format_route_status(alerts: list[AlertData]) -> list[RouteStatus]:
    """Convert raw alert data into RouteStatus list."""
    results: list[RouteStatus] = []
    for alert in alerts:
        route_type = _classify_route(alert.route_id)

        active_alerts: list[str] = []
        advisories: list[str] = []
        detours: list[str] = []

        if alert.current_message:
            active_alerts.append(_strip_html(alert.current_message))
        if alert.advisory_message:
            advisories.append(_strip_html(alert.advisory_message))
        if alert.detour_message:
            detours.append(_strip_html(alert.detour_message))

        is_normal = not active_alerts and not detours

        results.append(
            RouteStatus(
                route_name=alert.route_name,
                route_type=route_type,
                active_alerts=active_alerts,
                advisories=advisories,
                detours=detours,
                is_normal_service=is_normal,
                last_updated=alert.last_updated,
            )
        )
    return results


def _classify_route(route_id: str) -> str:
    """Determine route type from the alert route_id prefix."""
    rid = route_id.lower()
    if rid.startswith("rr_route_"):
        return "Regional Rail"
    if rid.startswith("bus_route_"):
        return "Bus"
    if rid.startswith("trolley_route_"):
        return "Trolley"
    if rid in ("bsl", "mfl"):
        return "Subway"
    if rid == "nhsl":
        return "Light Rail"
    return "Other"


def find_matching_routes(
    query: str, statuses: list[RouteStatus]
) -> list[RouteStatus]:
    """Find routes matching a user query (fuzzy)."""
    q = query.lower().strip()

    # Exact name match
    exact = [s for s in statuses if s.route_name.lower() == q]
    if exact:
        return exact

    # Substring match
    partial = [s for s in statuses if q in s.route_name.lower()]
    if partial:
        return partial

    # Fuzzy match
    scored = []
    for s in statuses:
        ratio = SequenceMatcher(None, q, s.route_name.lower()).ratio()
        if ratio > 0.55:
            scored.append((ratio, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:3]]


def format_rail_line_status(
    line_name: str,
    trains: list[TrainViewEntry],
    alerts: list[AlertData],
) -> RailLineStatus:
    """Build detailed status for a Regional Rail line."""
    line_trains = [t for t in trains if t.line and t.line.lower() == line_name.lower()]
    on_time = sum(1 for t in line_trains if t.late == 0)
    delayed = sum(1 for t in line_trains if t.late > 0)
    max_delay = max((t.late for t in line_trains), default=0)

    alert_msgs: list[str] = []
    for a in alerts:
        if a.current_message and line_name.lower() in a.route_name.lower():
            alert_msgs.append(_strip_html(a.current_message))

    train_details = [
        {
            "train": t.trainno,
            "destination": t.dest or "Unknown",
            "delay_minutes": t.late,
            "next_stop": t.nextstop,
            "track": t.TRACK,
        }
        for t in line_trains
    ]

    return RailLineStatus(
        line_name=line_name,
        active_trains=len(line_trains),
        trains_on_time=on_time,
        trains_delayed=delayed,
        max_delay_minutes=max_delay,
        alerts=alert_msgs,
        train_details=train_details,
    )


def format_bus_route_status(
    route: str,
    vehicles: list[TransitViewBus],
    alerts: list[AlertData],
) -> BusRouteStatus:
    """Build detailed status for a bus/trolley route."""
    delayed = [v for v in vehicles if v.late > 0]
    max_delay = max((v.late for v in vehicles), default=0)

    alert_msgs: list[str] = []
    detour_msgs: list[str] = []
    for a in alerts:
        if route.lower() in a.route_name.lower() or route in a.route_id:
            if a.current_message:
                alert_msgs.append(_strip_html(a.current_message))
            if a.detour_message:
                detour_msgs.append(_strip_html(a.detour_message))

    return BusRouteStatus(
        route_id=route,
        route_name=f"Route {route}",
        active_vehicles=len(vehicles),
        vehicles_with_delay=len(delayed),
        max_delay_minutes=max_delay,
        alerts=alert_msgs,
        detours=detour_msgs,
    )
