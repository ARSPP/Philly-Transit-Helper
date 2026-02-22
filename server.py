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
from septa_client import SeptaClient

# ---------------------------------------------------------------------------
# Lifespan: shared httpx.AsyncClient
# ---------------------------------------------------------------------------


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Create one httpx client for the entire server lifetime."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(HTTP_TIMEOUT),
        follow_redirects=True,
    ) as client:
        yield {"septa": SeptaClient(client)}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="SEPTA Transit Helper",
    instructions=(
        "Real-time SEPTA transit information for Philadelphia.\n\n"
        "TOOLS:\n"
        "- get_departure_options: plan trips between Regional Rail stations\n"
        "- station_departures: see upcoming trains at a station\n"
        "- check_status: check alerts and delays on any route\n\n"
        "DATA HONESTY:\n"
        "- All delay numbers come directly from SEPTA's API\n"
        "- If data is scheduled (not real-time), say so\n"
        "- If an API call fails, say so and show scheduled data only\n"
        "- Absence of alerts does NOT mean on-time service\n"
        "- Bus routes have real-time delay data per vehicle\n"
        "- Subway (BSL/MFL) has alerts only, no real-time train positions"
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
    ctx: Context = None,
) -> dict:
    """Plan ahead: find the next trains between two Regional Rail stations.

    Use when someone asks "When should I leave?" or "What are my options
    to get from A to B?" Returns ranked options with real-time delays,
    travel times, and recommendations.

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


@mcp.tool()
async def station_departures(
    station: Annotated[str, "Station name (e.g. 'Suburban Station', 'suburban', '30th st')"],
    count: Annotated[int, Field(description="Departures per direction", ge=1, le=20)] = DEFAULT_ARRIVALS_COUNT,
    direction: Annotated[str | None, "Filter: 'N' (northbound) or 'S' (southbound). Omit for both"] = None,
    ctx: Context = None,
) -> dict:
    """At the station now: see upcoming Regional Rail departures.

    Use when someone says "I'm at Suburban, what's coming?" or "What trains
    are leaving 30th Street?" Returns departures with destinations, lines,
    delays, and track numbers.
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        mcp.run(transport="http", host="0.0.0.0", port=8000)
    else:
        mcp.run(transport="stdio")
