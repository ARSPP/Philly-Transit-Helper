"""Pydantic models for SEPTA API responses and MCP tool outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# SEPTA API response models (parsing incoming JSON)
# ---------------------------------------------------------------------------

class NextToArriveEntry(BaseModel):
    """One trip from the NextToArrive endpoint."""

    orig_train: str
    orig_line: str | None = None
    orig_departure_time: str
    orig_arrival_time: str
    orig_delay: str | None = None
    isdirect: str | None = None  # "true" / "false" as string

    # Connection fields (present only for multi-leg trips)
    Connection: str | None = None
    term_train: str | None = None
    term_line: str | None = None
    term_depart_time: str | None = None
    term_arrival_time: str | None = None
    term_delay: str | None = None


class ArrivalEntry(BaseModel):
    """One train from the Arrivals endpoint."""

    direction: str | None = None
    path: str | None = None
    train_id: str | None = None
    origin: str | None = None
    destination: str | None = None
    line: str | None = None
    status: str | None = None
    service_type: str | None = None
    next_station: str | None = None
    sched_time: str | None = None
    depart_time: str | None = None
    track: str | None = None
    track_change: str | None = None
    platform: str | None = None
    platform_change: str | None = None


class TrainViewEntry(BaseModel):
    """One train from the TrainView endpoint."""

    lat: str | None = None
    lon: str | None = None
    trainno: str | None = None
    service: str | None = None
    dest: str | None = None
    currentstop: str | None = None
    nextstop: str | None = None
    line: str | None = None
    consist: str | None = None
    heading: str | None = None
    late: int = 0
    SOURCE: str | None = None
    TRACK: str | None = None
    TRACK_CHANGE: str | None = None


class TransitViewBus(BaseModel):
    """One bus/trolley from the TransitView endpoint."""

    lat: str | None = None
    lng: str | None = None
    label: str | None = None
    route_id: str | None = None
    trip: str | None = None
    VehicleID: str | None = None
    BlockID: str | None = None
    Direction: str | None = None
    destination: str | None = None
    heading: float | None = None
    late: int = 0
    next_stop_id: str | None = None
    next_stop_name: str | None = None
    next_stop_sequence: int | None = None
    estimated_seat_availability: str | None = None
    Offset: int | None = None
    Offset_sec: str | None = None
    timestamp: int | None = None


class AlertData(BaseModel):
    """One route's alert data from Alerts/get_alert_data.php."""

    route_id: str
    route_name: str
    current_message: str | None = None
    advisory_id: str | None = None
    advisory_message: str | None = None
    detour_message: str | None = None
    detour_id: str | None = None
    detour_start_location: str | None = None
    detour_start_date_time: str | None = None
    detour_end_date_time: str | None = None
    detour_reason: str | None = None
    last_updated: str | None = None
    isSnow: str | None = None


# ---------------------------------------------------------------------------
# MCP tool output models (what Claude sees)
# ---------------------------------------------------------------------------

class DepartureOption(BaseModel):
    """A single departure option returned by get_departure_options."""

    rank: int
    train_number: str
    line: str
    scheduled_departure: str
    scheduled_arrival: str
    delay_minutes: int | None = None
    delay_text: str
    is_direct: bool
    connection_info: str | None = None
    track: str | None = None
    travel_time_minutes: int | None = None
    arrives_by_target: bool | None = Field(
        None, description="Whether this train arrives before the arrive_by target"
    )
    backup_count: int | None = Field(
        None, description="Number of later trains that also arrive on time"
    )
    recommendation: str


class StationDeparture(BaseModel):
    """A single departure returned by station_departures."""

    direction: str
    train_number: str
    destination: str
    line: str | None = None
    scheduled_time: str
    delay_text: str
    delay_minutes: int | None = None
    track: str | None = None
    platform: str | None = None
    service_type: str | None = None


class RouteStatus(BaseModel):
    """Status for a route returned by check_status."""

    route_name: str
    route_type: str  # "Regional Rail", "Bus", "Trolley", "Subway"
    active_alerts: list[str] = Field(default_factory=list)
    advisories: list[str] = Field(default_factory=list)
    detours: list[str] = Field(default_factory=list)
    is_normal_service: bool
    last_updated: str | None = None


class BusRouteStatus(BaseModel):
    """Bus/trolley-specific status with vehicle info."""

    route_id: str
    route_name: str
    active_vehicles: int
    vehicles_with_delay: int
    max_delay_minutes: int
    alerts: list[str] = Field(default_factory=list)
    detours: list[str] = Field(default_factory=list)


class RailLineStatus(BaseModel):
    """Regional Rail line status with active train info."""

    line_name: str
    active_trains: int
    trains_on_time: int
    trains_delayed: int
    max_delay_minutes: int
    alerts: list[str] = Field(default_factory=list)
    train_details: list[dict] = Field(default_factory=list)
