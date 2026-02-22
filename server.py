"""SEPTA Real-Time MCP Server.

Provides three tools for Philadelphia transit information:
- get_departure_options: plan a trip between two Regional Rail stations
- station_departures: see what's coming at a station right now
- check_status: check alerts/delays on any SEPTA route
"""

from __future__ import annotations

import asyncio
import html as html_mod
import sys
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from string import Template
from typing import Annotated

import httpx
from fastmcp import Context, FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.websockets import WebSocket, WebSocketDisconnect

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

_rest_septa: SeptaClient | None = None


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Create one httpx client for the entire server lifetime."""
    global _rest_septa
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(HTTP_TIMEOUT),
        follow_redirects=True,
    ) as client:
        septa = SeptaClient(client)
        _rest_septa = septa
        yield {"septa": septa}
    _rest_septa = None


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
        "the user has a deadline.\n"
        "- station_departures: Use ONLY when someone is physically at a station and "
        "wants to see what's coming next ('I'm at 30th Street, what's leaving?').\n"
        "- check_status: Use for alerts, delays, and service disruptions on any route.\n\n"
        "RULES:\n"
        "- NEVER tell users to check the SEPTA app or Google Maps. You have real-time "
        "data — use it.\n"
        "- If the results don't cover the user's time window, call the tool again "
        "with count=10.\n"
        "- All delay numbers come directly from SEPTA's API.\n"
        "- If data is scheduled (not real-time), say so.\n"
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
    ctx: Context = None,
) -> dict:
    """Plan ahead: find the next trains between two Regional Rail stations.

    Use for ANY question about getting from A to B, including 'I need to
    arrive by X'. Always set arrive_by when the user mentions a deadline.
    Returns ranked options with real-time delays, travel times, and
    recommendations.

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
# Test endpoints (artifact sandbox connectivity test)
# ---------------------------------------------------------------------------

KOYEB_URL = "https://very-rozina-alexs-tools-d0f09f6b.koyeb.app"

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}


async def test_rest_handler(request: Request) -> Response:
    """REST endpoint with CORS — tests fetch() from artifact sandbox."""
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=_CORS_HEADERS)
    return JSONResponse(
        {
            "ok": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "fetch works!",
        },
        headers=_CORS_HEADERS,
    )


async def test_page_handler(request: Request) -> Response:
    """HTML page — tests iframe embedding from artifact sandbox."""
    html = """\
<!DOCTYPE html>
<html>
<head><title>SEPTA MCP Iframe Test</title></head>
<body style="font-family: system-ui; padding: 20px; background: #f8f9fa;">
    <h1 style="color: #2e7d32;">Iframe works!</h1>
    <p>If you can see this, the iframe loaded successfully from Koyeb.</p>
    <p id="clock" style="font-size: 1.2em; font-weight: bold;"></p>
    <script>
        setInterval(function() {
            document.getElementById('clock').textContent =
                'Current time: ' + new Date().toLocaleTimeString();
        }, 1000);
    </script>
</body>
</html>"""
    return HTMLResponse(html)


async def test_ws_handler(websocket: WebSocket) -> None:
    """WebSocket endpoint — tests WS connectivity from artifact sandbox."""
    await websocket.accept()
    tick = 0
    try:
        while True:
            tick += 1
            await websocket.send_json(
                {
                    "tick": tick,
                    "time": datetime.now(timezone.utc).isoformat(),
                }
            )
            await asyncio.sleep(3)
    except (WebSocketDisconnect, Exception):
        pass


# ---------------------------------------------------------------------------
# Tool 4: artifact_sandbox_test
# ---------------------------------------------------------------------------

_SANDBOX_TEST_COMPONENT = """\
import { useState, useEffect } from 'react';

export default function SandboxTest() {
  const SERVER = '""" + KOYEB_URL + """';
  const WS_SERVER = SERVER.replace('https://', 'wss://');

  const [fetchResult, setFetchResult] = useState('Testing...');
  const [wsResult, setWsResult] = useState('Testing...');
  const [iframeStatus, setIframeStatus] = useState('Waiting...');

  useEffect(() => {
    // Test 1: fetch()
    fetch(SERVER + '/api/test')
      .then(r => r.json())
      .then(data => setFetchResult('SUCCESS: ' + JSON.stringify(data)))
      .catch(err => setFetchResult('BLOCKED: ' + err.message));

    // Test 3: WebSocket
    try {
      const ws = new WebSocket(WS_SERVER + '/ws/test');
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        setWsResult('SUCCESS: tick=' + data.tick + ' time=' + data.time);
      };
      ws.onerror = () => setWsResult('BLOCKED: WebSocket connection failed');
      ws.onclose = (e) => setWsResult(prev =>
        prev.startsWith('SUCCESS') ? prev : 'BLOCKED: WebSocket closed (code=' + e.code + ')'
      );
      return () => ws.close();
    } catch (err) {
      setWsResult('BLOCKED: ' + err.message);
    }
  }, []);

  const boxStyle = (result) => ({
    padding: '12px 16px',
    borderRadius: 8,
    fontFamily: 'monospace',
    fontSize: 14,
    marginBottom: 8,
    background: result.startsWith('SUCCESS') ? '#e8f5e9' : result === 'Testing...' || result === 'Waiting...' ? '#fff3e0' : '#ffebee',
    border: '1px solid ' + (result.startsWith('SUCCESS') ? '#4caf50' : result === 'Testing...' || result === 'Waiting...' ? '#ff9800' : '#f44336'),
  });

  return (
    <div style={{ fontFamily: 'system-ui', padding: 24, maxWidth: 700 }}>
      <h2 style={{ marginTop: 0 }}>Artifact Sandbox Connectivity Test</h2>
      <p style={{ color: '#666' }}>Server: {SERVER}</p>

      <h3>Test 1: fetch()</h3>
      <div style={boxStyle(fetchResult)}>{fetchResult}</div>

      <h3>Test 2: iframe</h3>
      <iframe
        src={SERVER + '/test-page'}
        style={{ width: '100%', height: 120, border: '1px solid #ccc', borderRadius: 8 }}
        onLoad={() => setIframeStatus('SUCCESS: iframe onLoad fired')}
        onError={() => setIframeStatus('BLOCKED: iframe failed to load')}
      />
      <div style={boxStyle(iframeStatus)}>{iframeStatus}</div>

      <h3>Test 3: WebSocket</h3>
      <div style={boxStyle(wsResult)}>{wsResult}</div>

      <hr style={{ margin: '24px 0', border: 'none', borderTop: '1px solid #ddd' }} />
      <p style={{ color: '#999', fontSize: 12 }}>
        This artifact tests whether Claude's sandbox CSP allows network calls to external servers.
        Each test shows SUCCESS (green) or BLOCKED (red).
      </p>
    </div>
  );
}
"""


@mcp.tool()
async def artifact_sandbox_test() -> dict:
    """Returns a React component that tests network connectivity from Claude's artifact sandbox.

    Call this tool and render the returned component as a React artifact.
    It tests fetch(), iframe, and WebSocket connections to our server.
    """
    return {
        "artifact_type": "application/vnd.ant.react",
        "title": "Sandbox Connectivity Test",
        "component_source": _SANDBOX_TEST_COMPONENT,
        "instructions": (
            "Render the component_source above as a React artifact. "
            "It tests whether fetch, iframe, and WebSocket connections "
            "to our Koyeb server work from Claude's artifact sandbox."
        ),
    }


# ---------------------------------------------------------------------------
# Live departures board (iframe-based)
# ---------------------------------------------------------------------------

_BOARD_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="35">
<title>SEPTA - $station</title>
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
.empty{text-align:center;padding:40px;color:#718096;font-size:14px}
.err{text-align:center;padding:40px;color:#fc8181;font-size:14px}
@media(max-width:500px){td,th{padding:6px 5px;font-size:12px}th{font-size:9px}.line{font-size:11px}}
</style>
</head>
<body>
<div class="hdr">
<h1>$station</h1>
<span class="t">$time</span>
</div>
$content
<div class="ft"><span>Auto-refreshes every 30s</span><span>SEPTA Real-Time</span></div>
<script>setTimeout(function(){location.reload()},30000)</script>
</body>
</html>""")


def _render_board(station: str, departures: list) -> str:
    """Render a full HTML departures board page."""
    station_safe = html_mod.escape(station)
    time_str = datetime.now().strftime("%I:%M:%S %p")

    if not departures:
        content = '<div class="empty">No upcoming departures found.<br>Service may have ended for the day.</div>'
        return _BOARD_TEMPLATE.substitute(
            station=station_safe, time=time_str, content=content
        )

    rows: list[str] = []
    for d in departures:
        delay = d.delay_minutes
        if delay is not None and delay == 0:
            cls = "g"
        elif delay is not None and delay <= 5:
            cls = "y"
        elif delay is not None and delay > 5:
            cls = "r"
        else:
            cls = "s"

        arrow = ""
        if d.direction == "N":
            arrow = '<span class="dir">&#x2191;</span>'
        elif d.direction == "S":
            arrow = '<span class="dir">&#x2193;</span>'

        rows.append(
            f"<tr>"
            f'<td>{html_mod.escape(d.scheduled_time or "")}</td>'
            f'<td class="dest">{html_mod.escape(d.destination or "")}{arrow}</td>'
            f'<td class="line">{html_mod.escape(d.line or "")}</td>'
            f'<td class="{cls}">{html_mod.escape(d.delay_text or "Scheduled")}</td>'
            f'<td class="trk">{html_mod.escape(d.track or "—")}</td>'
            f"</tr>"
        )

    content = (
        "<table><thead><tr>"
        "<th>Time</th><th>Destination</th><th>Line</th><th>Status</th><th>Trk</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return _BOARD_TEMPLATE.substitute(
        station=station_safe, time=time_str, content=content
    )


def _render_board_error(message: str) -> str:
    """Render a board page with an error message."""
    return _BOARD_TEMPLATE.substitute(
        station="SEPTA Departures",
        time=datetime.now().strftime("%I:%M:%S %p"),
        content=f'<div class="err">{html_mod.escape(message)}</div>',
    )


async def board_handler(request: Request) -> Response:
    """Server-rendered live departures board page."""
    station = request.query_params.get("station", "")
    if not station:
        return HTMLResponse(_render_board_error("No station specified. Use ?station=Suburban+Station"))

    if _rest_septa is None:
        return HTMLResponse(_render_board_error("Server starting up..."), status_code=503)

    resolved, _ = resolve_station_name(station)
    direction = request.query_params.get("direction")

    try:
        arrivals = await _rest_septa.get_station_arrivals(resolved, 10, direction)
    except Exception:
        return HTMLResponse(_render_board_error(f"Could not reach SEPTA API for {resolved}"))

    train_view = None
    try:
        train_view = await _rest_septa.get_train_view()
    except Exception:
        pass

    departures = format_station_departures(arrivals, train_view)
    return HTMLResponse(_render_board(resolved, departures))


# ---------------------------------------------------------------------------
# Tool 5: live_board
# ---------------------------------------------------------------------------


@mcp.tool()
async def live_board(
    station: Annotated[str, "Station name (e.g. 'Suburban Station', 'suburban', '30th st')"],
    ctx: Context = None,
) -> dict:
    """Show a live, auto-refreshing departures board for a station.

    Returns an iframe-based artifact that displays real-time departures
    with delay status, track numbers, and auto-refreshes every 30 seconds.
    Render the component_source as a React artifact.
    """
    resolved, was_exact = resolve_station_name(station)
    board_url = f"{KOYEB_URL}/board?station={urllib.parse.quote(resolved)}"

    notes: list[str] = []
    if not was_exact:
        notes.append(f"Interpreted '{station}' as '{resolved}'")

    component = (
        "export default function DepartureBoard() {\n"
        "  return (\n"
        "    <iframe\n"
        f'      src="{board_url}"\n'
        '      style={{ width: "100%", height: "100%", border: "none" }}\n'
        f'      title="SEPTA Departures - {resolved}"\n'
        "    />\n"
        "  );\n"
        "}\n"
    )

    result: dict = {
        "station": resolved,
        "board_url": board_url,
        "component_source": component,
        "instructions": (
            "Render the component_source as a React artifact to show the live "
            "departures board. The board auto-refreshes every 30 seconds with "
            "real-time SEPTA data. If the artifact iframe doesn't render, provide "
            "the board_url as a direct link the user can open."
        ),
    }
    if notes:
        result["notes"] = notes
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route, WebSocketRoute

        import uvicorn

        mcp_app = mcp.http_app()

        app = Starlette(
            routes=[
                Route("/board", board_handler),
                Route("/api/test", test_rest_handler, methods=["GET", "OPTIONS"]),
                Route("/test-page", test_page_handler),
                WebSocketRoute("/ws/test", test_ws_handler),
                Mount("/", mcp_app),
            ],
            lifespan=mcp_app.lifespan,
        )

        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        mcp.run(transport="stdio")
