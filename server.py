"""SEPTA Real-Time MCP Server.

Provides three tools for Philadelphia transit information:
- get_departure_options: plan a trip between two Regional Rail stations
- station_departures: see what's coming at a station right now
- check_status: check alerts/delays on any SEPTA route
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastmcp import Context, FastMCP
from pydantic import Field

from config import (
    DEFAULT_ARRIVALS_COUNT,
    DEFAULT_DEPARTURE_COUNT,
    HTTP_TIMEOUT,
    LINE_TO_ALERT_ID,
    REGIONAL_RAIL_LINES,
    TRANSIT_ALERT_IDS,
)
from departure_logic import (
    find_matching_routes,
    format_bus_route_status,
    format_rail_line_status,
    format_route_status,
    format_station_departures,
    rank_departure_options,
    resolve_station_name,
)
from gtfs_loader import GtfsData, load_gtfs
from schedule_logic import (
    lookup_schedule,
    lookup_trip_schedule,
    parse_date,
    parse_time_to_seconds,
)
from septa_client import SeptaClient

# ---------------------------------------------------------------------------
# Lifespan: shared httpx.AsyncClient
# ---------------------------------------------------------------------------


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Create one httpx client and load GTFS for the entire server lifetime."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(HTTP_TIMEOUT),
        follow_redirects=True,
    ) as client:
        septa = SeptaClient(client)
        try:
            gtfs = await load_gtfs(client)
        except Exception:
            gtfs = None  # schedule tools degrade gracefully
        yield {"septa": septa, "gtfs": gtfs}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="SEPTA Transit Helper",
    instructions=(
        "Real-time SEPTA transit information for Philadelphia.\n\n"
        "TOOL SELECTION:\n"
        "- get_departure_options: Use for ANY trip-planning question ('how do I get "
        "from A to B?', 'I need to be at X by 3 PM'). Always set arrive_by when "
        "the user has a deadline. Accepts an optional 'date' param for future dates "
        "('tomorrow', 'next Monday', '2026-03-01').\n"
        "- station_departures: Use ONLY when someone is physically at a station and "
        "wants to see what's coming next ('I'm at 30th Street, what's leaving?').\n"
        "- schedule_lookup: Use for schedule questions about future dates "
        "('what trains leave Suburban tomorrow morning?', 'weekend schedule at "
        "30th Street'). Returns published timetable data.\n"
        "- check_status: Use for alerts, delays, and service disruptions on any route.\n\n"
        "RULES:\n"
        "- NEVER tell users to check the SEPTA app or Google Maps. You have real-time "
        "data — use it.\n"
        "- If the results don't cover the user's time window, call the tool again "
        "with count=10.\n"
        "- All delay numbers come directly from SEPTA's API.\n"
        "- If data is scheduled (not real-time), say so — this always applies to "
        "future-date results from schedule_lookup and get_departure_options with date.\n"
        "- If an API call fails, say so and show scheduled data only.\n"
        "- Absence of alerts does NOT mean on-time service.\n"
        "- Bus routes have real-time delay data per vehicle.\n"
        "- Subway (BSL/MFL) has alerts only, no real-time train positions."
    ),
    lifespan=app_lifespan,
)


# ---------------------------------------------------------------------------
# Tool 1: get_departure_options
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_departure_options(
    origin: Annotated[str, "Starting station (e.g. 'Suburban Station', 'suburban', '30th st')"],
    destination: Annotated[str, "Ending station (e.g. 'Paoli', 'Thorndale', 'airport b')"],
    count: Annotated[int, Field(description="Number of options to show", ge=1, le=10)] = DEFAULT_DEPARTURE_COUNT,
    arrive_by: Annotated[str | None, "Target arrival time (e.g. '3:00 PM', '3:00PM'). Omit to just show next departures"] = None,
    date: Annotated[str | None, "Date for the trip (e.g. 'tomorrow', 'next Monday', '2026-03-01'). Omit for today/real-time"] = None,
    ctx: Context = None,
) -> dict:
    """Plan ahead: find the next trains between two Regional Rail stations.

    Use for ANY question about getting from A to B, including 'I need to
    arrive by X'. Always set arrive_by when the user mentions a deadline.

    For today: returns ranked options with real-time delays.
    For future dates: returns published schedule data (set the date param).

    Station names are fuzzy-matched — 'suburban', '30th st', 'market east'
    all work.
    """
    septa: SeptaClient = ctx.lifespan_context["septa"]

    resolved_origin, origin_exact = resolve_station_name(origin)
    resolved_dest, dest_exact = resolve_station_name(destination)

    notes: list[str] = []
    if not origin_exact:
        notes.append(f"Interpreted '{origin}' as '{resolved_origin}'")
    if not dest_exact:
        notes.append(f"Interpreted '{destination}' as '{resolved_dest}'")

    # --- Future date: use GTFS schedule data ---
    if date is not None:
        import datetime as _dt

        query_date = parse_date(date)
        if query_date is None:
            return {"error": f"Could not parse date: '{date}'. Try 'tomorrow', 'next Monday', or 'YYYY-MM-DD'."}

        today = _dt.date.today()
        if query_date <= today:
            # Fall through to real-time for today/past dates
            pass
        else:
            gtfs: GtfsData | None = ctx.lifespan_context.get("gtfs")
            if gtfs is None:
                return {"error": "Schedule data is not available. Try again later."}

            time_after = parse_time_to_seconds(arrive_by) if arrive_by else None
            trips = lookup_trip_schedule(
                gtfs, resolved_origin, resolved_dest, query_date,
                time_after=time_after, count=count,
            )
            result: dict = {
                "origin": resolved_origin,
                "destination": resolved_dest,
                "date": query_date.strftime("%A, %B %-d, %Y"),
                "data_freshness": "Published schedule (not real-time — delays not included)",
                "options": [t.model_dump(exclude_none=True) for t in trips],
            }
            if not trips:
                result["message"] = f"No scheduled trains found from {resolved_origin} to {resolved_dest} on {result['date']}."
            if notes:
                result["notes"] = notes
            return result

    try:
        entries = await septa.get_next_to_arrive(
            resolved_origin, resolved_dest, count
        )
    except httpx.HTTPStatusError as e:
        return {
            "error": f"SEPTA API error ({e.response.status_code}). Station names may be invalid.",
            "tried_origin": resolved_origin,
            "tried_destination": resolved_dest,
            "notes": notes,
        }
    except httpx.RequestError:
        return {
            "error": "Could not reach SEPTA API. Try again shortly.",
            "tried_origin": resolved_origin,
            "tried_destination": resolved_dest,
            "notes": notes,
        }

    if not entries:
        return {
            "origin": resolved_origin,
            "destination": resolved_dest,
            "options": [],
            "message": "No upcoming trains found for this route right now.",
            "notes": notes,
        }

    # Fetch TrainView for track numbers and fresher delay data
    train_view = None
    try:
        train_view = await septa.get_train_view()
    except Exception:
        notes.append("Could not fetch real-time train positions — track numbers unavailable")

    options = rank_departure_options(entries, arrive_by=arrive_by, train_view=train_view)

    result: dict = {
        "origin": resolved_origin,
        "destination": resolved_dest,
        "data_freshness": "Real-time delays from SEPTA — may change",
    }

    if arrive_by:
        viable = [o for o in options if o.arrives_by_target is True]
        late = [o for o in options if o.arrives_by_target is not True]
        result["arrive_by_target"] = arrive_by
        result["options"] = [opt.model_dump(exclude_none=True) for opt in viable]
        result["options_meeting_target"] = len(viable)
        if late:
            result["later_alternatives"] = [
                opt.model_dump(exclude_none=True) for opt in late
            ]
    else:
        result["options"] = [opt.model_dump(exclude_none=True) for opt in options]

    if notes:
        result["notes"] = notes

    return result


# ---------------------------------------------------------------------------
# Tool 2: station_departures
# ---------------------------------------------------------------------------


@mcp.tool(app={"resourceUri": "ui://departures/board.html"})
async def station_departures(
    station: Annotated[str, "Station name (e.g. 'Suburban Station', 'suburban', '30th st')"],
    count: Annotated[int, Field(description="Departures per direction", ge=1, le=20)] = DEFAULT_ARRIVALS_COUNT,
    direction: Annotated[str | None, "Filter: 'N' (northbound) or 'S' (southbound). Omit for both"] = None,
    ctx: Context = None,
) -> dict:
    """At the station now: see upcoming Regional Rail departures.

    Use when someone says "I'm at Suburban, what's coming?" or "What trains
    are leaving 30th Street?" Returns departures with destinations, lines,
    delays, and track numbers. Also renders a live departures board UI.
    """
    septa: SeptaClient = ctx.lifespan_context["septa"]

    resolved, was_exact = resolve_station_name(station)
    notes: list[str] = []
    if not was_exact:
        notes.append(f"Interpreted '{station}' as '{resolved}'")

    try:
        arrivals = await septa.get_station_arrivals(resolved, count, direction)
    except httpx.HTTPStatusError as e:
        return {
            "error": f"SEPTA API error ({e.response.status_code}). Station name may be invalid.",
            "tried_station": resolved,
            "notes": notes,
        }
    except httpx.RequestError:
        return {
            "error": "Could not reach SEPTA API. Try again shortly.",
            "tried_station": resolved,
            "notes": notes,
        }

    # TrainView for real-time enrichment (delay/track updates)
    train_view = None
    try:
        train_view = await septa.get_train_view()
    except Exception:
        notes.append("Could not fetch real-time train positions — showing scheduled data")

    departures = format_station_departures(arrivals, train_view)

    if not departures:
        return {
            "station": resolved,
            "departures": [],
            "message": "No upcoming departures found. Service may have ended for the day.",
            "notes": notes,
        }

    result: dict = {
        "station": resolved,
        "departures": [d.model_dump(exclude_none=True) for d in departures],
        "data_freshness": "Real-time, updates every 30 seconds",
    }
    if notes:
        result["notes"] = notes
    return result


# ---------------------------------------------------------------------------
# Tool 3: check_status
# ---------------------------------------------------------------------------


@mcp.tool()
async def check_status(
    route: Annotated[str | None, "Route or line name (e.g. 'Paoli/Thorndale', '21', 'BSL'). Omit for system-wide alerts"] = None,
    ctx: Context = None,
) -> dict:
    """Check service status, alerts, and delays for any SEPTA route.

    Use when someone asks "How's the Paoli/Thorndale?" or "Any alerts on
    the BSL?" or "Any service disruptions?"

    Works for Regional Rail, buses, trolleys, and subway.
    System-wide if no route specified.
    """
    septa: SeptaClient = ctx.lifespan_context["septa"]

    try:
        all_alerts = await septa.get_alerts()
    except httpx.RequestError:
        return {"error": "Could not reach SEPTA API. Try again shortly."}

    all_statuses = format_route_status(all_alerts)

    # --- System-wide (no route specified) ---
    if not route:
        issues = [s for s in all_statuses if not s.is_normal_service]
        return {
            "query": "System-wide",
            "total_routes_monitored": len(all_statuses),
            "routes_with_issues": len(issues),
            "alerts": [s.model_dump(exclude_none=True) for s in issues],
            "message": (
                f"{len(issues)} route(s) have active alerts or detours."
                if issues
                else "No active alerts reported. Note: this does not confirm all services are on time."
            ),
        }

    # --- Specific route ---
    route_lower = route.lower().strip()

    # Check if it's a Regional Rail line
    matching_rr_line = _match_rail_line(route_lower)
    if matching_rr_line:
        return await _rail_line_status(septa, matching_rr_line, all_alerts)

    # Check if it's a subway/NHSL
    for key, name in TRANSIT_ALERT_IDS.items():
        if route_lower in (key, name.lower()):
            matching = [s for s in all_statuses if s.route_name.lower() == name.lower()]
            if not matching:
                matching = [s for s in all_statuses if key in s.route_name.lower() or name.lower() in s.route_name.lower()]
            return {
                "query": name,
                "route_type": "Subway" if key in ("bsl", "mfl") else "Light Rail",
                "status": [s.model_dump(exclude_none=True) for s in matching],
                "note": "No real-time train positions available for subway/light rail. Alert status only.",
            }

    # Check if it's a bus/trolley route number
    if route_lower.isdigit() or route_lower.startswith("route "):
        route_num = route_lower.replace("route ", "").strip()
        return await _bus_route_status(septa, route_num, all_alerts)

    # Fuzzy match against all alert route names
    matching = find_matching_routes(route, all_statuses)
    if matching:
        return {
            "query": route,
            "results": [s.model_dump(exclude_none=True) for s in matching],
        }

    return {
        "query": route,
        "results": [],
        "message": f"No route found matching '{route}'. Try the full line name (e.g. 'Paoli/Thorndale') or route number (e.g. '21').",
    }


def _match_rail_line(query: str) -> str | None:
    """Match a query to a Regional Rail line name."""
    for line in REGIONAL_RAIL_LINES:
        if query == line.lower():
            return line
        if query in line.lower():
            return line
    # Common abbreviations
    abbrevs = {
        "paoli": "Paoli/Thorndale",
        "thorndale": "Paoli/Thorndale",
        "media": "Media/Elwyn",
        "elwyn": "Media/Elwyn",
        "lansdale": "Lansdale/Doylestown",
        "doylestown": "Lansdale/Doylestown",
        "manayunk": "Manayunk/Norristown",
        "norristown": "Manayunk/Norristown",
        "chestnut hill east": "Chestnut Hill East",
        "chestnut hill west": "Chestnut Hill West",
        "che": "Chestnut Hill East",
        "chw": "Chestnut Hill West",
        "wilmington": "Wilmington/Newark",
        "west trenton": "West Trenton",
    }
    return abbrevs.get(query)


async def _rail_line_status(
    septa: SeptaClient,
    line_name: str,
    alerts: list,
) -> dict:
    """Build detailed status for a Regional Rail line."""
    train_view = []
    try:
        train_view = await septa.get_train_view()
    except Exception:
        pass

    status = format_rail_line_status(line_name, train_view, alerts)
    result = status.model_dump(exclude_none=True)
    result["route_type"] = "Regional Rail"
    result["query"] = line_name
    if not train_view:
        result["note"] = "Could not fetch real-time train data. Alert info only."
    return result


async def _bus_route_status(
    septa: SeptaClient,
    route_num: str,
    alerts: list,
) -> dict:
    """Build detailed status for a bus/trolley route."""
    vehicles = []
    try:
        vehicles = await septa.get_transit_view(route_num)
    except Exception:
        pass

    status = format_bus_route_status(route_num, vehicles, alerts)
    result = status.model_dump(exclude_none=True)
    result["route_type"] = "Bus/Trolley"
    result["query"] = f"Route {route_num}"
    if not vehicles:
        result["note"] = "No active vehicles found. Route may not be running right now."
    return result


# ---------------------------------------------------------------------------
# Tool 4: schedule_lookup
# ---------------------------------------------------------------------------


@mcp.tool()
async def schedule_lookup(
    station: Annotated[str, "Station name (e.g. 'Suburban Station', 'suburban', '30th st')"],
    date: Annotated[str, "Date to look up (e.g. 'tomorrow', 'next Monday', '2026-03-01')"],
    time_after: Annotated[str | None, "Show departures after this time (e.g. '8:00 AM')"] = None,
    time_before: Annotated[str | None, "Show departures before this time (e.g. '9:00 AM')"] = None,
    count: Annotated[int, Field(description="Number of departures to show", ge=1, le=30)] = 10,
    ctx: Context = None,
) -> dict:
    """Look up the published train schedule for any future date.

    Use when someone asks about tomorrow's trains, next week's schedule,
    or any date beyond today. Returns the published SEPTA timetable.

    For today's real-time data, use station_departures or
    get_departure_options instead.
    """
    gtfs: GtfsData | None = ctx.lifespan_context.get("gtfs")
    if gtfs is None:
        return {"error": "Schedule data is not available. Try again later."}

    query_date = parse_date(date)
    if query_date is None:
        return {"error": f"Could not parse date: '{date}'. Try 'tomorrow', 'next Monday', or 'YYYY-MM-DD'."}

    resolved, was_exact = resolve_station_name(station)
    notes: list[str] = []
    if not was_exact:
        notes.append(f"Interpreted '{station}' as '{resolved}'")

    after_secs = parse_time_to_seconds(time_after) if time_after else None
    before_secs = parse_time_to_seconds(time_before) if time_before else None

    departures = lookup_schedule(
        gtfs, resolved, query_date,
        time_after=after_secs,
        time_before=before_secs,
        count=count,
    )

    result: dict = {
        "station": resolved,
        "date": query_date.strftime("%A, %B %-d, %Y"),
        "data_freshness": "Published schedule (not real-time — delays not included)",
        "departures": [d.model_dump(exclude_none=True) for d in departures],
    }

    if time_after:
        result["time_filter_after"] = time_after
    if time_before:
        result["time_filter_before"] = time_before

    if not departures:
        result["message"] = f"No scheduled departures found at {resolved} on {result['date']}."

    if notes:
        result["notes"] = notes

    return result


# ---------------------------------------------------------------------------
# Tool 5: poll_departures (app-only — called by the departures board UI)
# ---------------------------------------------------------------------------


@mcp.tool(app={"visibility": ["app"]})
async def poll_departures(
    station: Annotated[str, "Station name"],
    direction: Annotated[str | None, "Filter: 'N' or 'S'. Omit for both"] = None,
    ctx: Context = None,
) -> dict:
    """Poll fresh departure data (called by the departures board UI every 30s)."""
    septa: SeptaClient = ctx.lifespan_context["septa"]

    resolved, _ = resolve_station_name(station)

    try:
        arrivals = await septa.get_station_arrivals(resolved, 10, direction)
    except Exception:
        return {"error": "Could not reach SEPTA API", "station": resolved}

    train_view = None
    try:
        train_view = await septa.get_train_view()
    except Exception:
        pass

    departures = format_station_departures(arrivals, train_view)
    return {
        "station": resolved,
        "departures": [d.model_dump(exclude_none=True) for d in departures],
    }


# ---------------------------------------------------------------------------
# MCP App UI: departures board (served as ui:// resource)
# ---------------------------------------------------------------------------

_BOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SEPTA Departures</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f1923;color:#e0e0e0;font-family:-apple-system,system-ui,'Segoe UI',sans-serif;padding:16px;min-height:100vh}
.hdr{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;padding-bottom:10px;border-bottom:2px solid #2d3748}
.hdr h1{font-size:17px;color:#fff;font-weight:600}
.hdr .t{font-size:11px;color:#718096}
table{width:100%;border-collapse:collapse}
th{text-align:left;padding:6px 10px;font-size:10px;color:#718096;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid #2d3748}
td{padding:8px 10px;font-size:13px;border-bottom:1px solid #1a2332}
tr:nth-child(even){background:#131f2e}
.g{color:#48bb78}.y{color:#f6ad55}.r{color:#fc8181}.s{color:#a0aec0}
.trk{font-weight:700;color:#63b3ed;text-align:center;min-width:28px}
.dir{font-size:10px;color:#a0aec0;margin-left:4px}
.dest{white-space:nowrap}
.line{color:#a0aec0;font-size:12px}
.ft{margin-top:14px;padding-top:10px;border-top:1px solid #2d3748;font-size:10px;color:#4a5568;display:flex;justify-content:space-between}
.loading{text-align:center;padding:40px;color:#718096;font-size:14px}
.err{text-align:center;padding:40px;color:#fc8181;font-size:14px}
@media(max-width:500px){td,th{padding:6px 5px;font-size:12px}th{font-size:9px}.line{font-size:11px}}
</style>
</head>
<body>
<div class="hdr">
<h1 id="station-name">SEPTA Departures</h1>
<span class="t" id="updated"></span>
</div>
<div id="content"><div class="loading">Waiting for data...</div></div>
<div class="ft"><span>Auto-refreshes every 30s</span><span>SEPTA Real-Time</span></div>

<script type="module">
import { App } from "https://unpkg.com/@modelcontextprotocol/ext-apps@0.4.0/app-with-deps";

let currentStation = null;

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function parseResult(result) {
  const text = (result.content || []).find(c => c.type === "text");
  if (text && text.text) {
    try { return JSON.parse(text.text); } catch(e) { return null; }
  }
  return null;
}

function render(data) {
  if (data.error) {
    document.getElementById("content").innerHTML =
      '<div class="err">' + esc(data.error) + '</div>';
    return;
  }

  currentStation = data.station;
  document.getElementById("station-name").textContent = data.station;
  document.getElementById("updated").textContent =
    "Updated " + new Date().toLocaleTimeString();

  const deps = data.departures || [];
  if (deps.length === 0) {
    document.getElementById("content").innerHTML =
      '<div class="loading">No upcoming departures found.</div>';
    return;
  }

  let html = '<table><thead><tr>' +
    '<th>Time</th><th>Destination</th><th>Line</th><th>Status</th><th>Trk</th>' +
    '</tr></thead><tbody>';

  for (const d of deps) {
    const dm = d.delay_minutes;
    const cls = dm != null && dm === 0 ? "g"
      : dm != null && dm <= 5 ? "y"
      : dm != null && dm > 5 ? "r" : "s";
    const arrow = d.direction === "N" ? " \\u2191"
      : d.direction === "S" ? " \\u2193" : "";

    html += '<tr>' +
      '<td>' + esc(d.scheduled_time || "") + '</td>' +
      '<td class="dest">' + esc(d.destination || "") +
        '<span class="dir">' + arrow + '</span></td>' +
      '<td class="line">' + esc(d.line || "") + '</td>' +
      '<td class="' + cls + '">' + esc(d.delay_text || "Scheduled") + '</td>' +
      '<td class="trk">' + esc(d.track || "\\u2014") + '</td>' +
      '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById("content").innerHTML = html;
}

// --- MCP Apps SDK ---
const app = new App({ name: "SEPTA Departures Board", version: "1.0.0" });

// Set handlers BEFORE connect (per SDK best practices)
app.ontoolresult = (result) => {
  const data = parseResult(result);
  if (data) render(data);
};

app.onerror = console.error;

await app.connect();

// Auto-poll every 30 seconds
setInterval(async () => {
  if (!currentStation) return;
  try {
    const result = await app.callServerTool({
      name: "poll_departures",
      arguments: { station: currentStation }
    });
    const data = parseResult(result);
    if (data) render(data);
  } catch(e) { /* retry next cycle */ }
}, 30000);
</script>
</body>
</html>
"""


@mcp.resource(
    "ui://departures/board.html",
    app={"csp": {"resourceDomains": ["https://unpkg.com"]}},
)
async def departures_board_ui() -> str:
    """Departures board MCP App UI."""
    return _BOARD_HTML


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        mcp.run(transport="http", host="0.0.0.0", port=8000, stateless_http=True)
    else:
        mcp.run(transport="stdio")
