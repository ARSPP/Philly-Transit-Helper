"""Async HTTP client wrapping all SEPTA API endpoints."""

from __future__ import annotations

import logging

import httpx
from pydantic import ValidationError

from cache import cached_async
from config import (
    ALERTS_URL,
    ARRIVALS_URL,
    BUS_DETOURS_URL,
    CACHE_TTL_ALERTS,
    CACHE_TTL_ARRIVALS,
    CACHE_TTL_NEXT_TO_ARRIVE,
    CACHE_TTL_TRAIN_VIEW,
    CACHE_TTL_TRANSIT_VIEW,
    DEFAULT_ARRIVALS_COUNT,
    DEFAULT_DEPARTURE_COUNT,
    NEXT_TO_ARRIVE_URL,
    TRAIN_VIEW_URL,
    TRANSIT_VIEW_URL,
)
from models import (
    AlertData,
    ArrivalEntry,
    NextToArriveEntry,
    TrainViewEntry,
    TransitViewBus,
)

log = logging.getLogger(__name__)


def _safe_parse(model_cls, items):
    """Parse a list of dicts into Pydantic models, skipping bad entries."""
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            results.append(model_cls.model_validate(item))
        except (ValidationError, Exception):
            log.debug("Skipping unparseable %s entry: %s", model_cls.__name__, item)
            continue
    return results


class SeptaClient:
    """Async wrapper around SEPTA's public REST APIs.

    Each method corresponds to one SEPTA endpoint, returns parsed Pydantic
    models, and is cached with a per-endpoint TTL.

    Parsing is defensive: bad entries are skipped rather than crashing.
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self.client = http_client

    # ------------------------------------------------------------------
    # Regional Rail
    # ------------------------------------------------------------------

    @cached_async(ttl=CACHE_TTL_NEXT_TO_ARRIVE)
    async def get_next_to_arrive(
        self,
        origin: str,
        destination: str,
        count: int = DEFAULT_DEPARTURE_COUNT,
    ) -> list[NextToArriveEntry]:
        """Next trains between two Regional Rail stations."""
        resp = await self.client.get(
            NEXT_TO_ARRIVE_URL,
            params={"req1": origin, "req2": destination, "req3": count},
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        return _safe_parse(NextToArriveEntry, data)

    @cached_async(ttl=CACHE_TTL_ARRIVALS)
    async def get_station_arrivals(
        self,
        station: str,
        count: int = DEFAULT_ARRIVALS_COUNT,
        direction: str | None = None,
    ) -> dict[str, list[ArrivalEntry]]:
        """Upcoming arrivals/departures at a station.

        Returns ``{"Northbound": [...], "Southbound": [...]}``.

        The SEPTA Arrivals endpoint nests data under a dynamic key
        (station name + timestamp) containing a list of direction dicts.
        """
        params: dict[str, str | int] = {"station": station, "results": count}
        if direction:
            params["direction"] = direction

        resp = await self.client.get(ARRIVALS_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        result: dict[str, list[ArrivalEntry]] = {
            "Northbound": [],
            "Southbound": [],
        }

        if not isinstance(data, dict):
            return result

        # data is {"Station Name Departures: date": [{dir: [...]}, ...]}
        for _station_key, direction_list in data.items():
            if not isinstance(direction_list, list):
                continue
            for direction_dict in direction_list:
                if not isinstance(direction_dict, dict):
                    continue
                for dir_name, entries in direction_dict.items():
                    if not isinstance(entries, list):
                        continue
                    parsed = _safe_parse(ArrivalEntry, entries)
                    if dir_name in result:
                        result[dir_name].extend(parsed)
                    else:
                        result[dir_name] = parsed

        return result

    @cached_async(ttl=CACHE_TTL_TRAIN_VIEW)
    async def get_train_view(self) -> list[TrainViewEntry]:
        """All active Regional Rail trains system-wide."""
        resp = await self.client.get(TRAIN_VIEW_URL)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        return _safe_parse(TrainViewEntry, data)

    # ------------------------------------------------------------------
    # Bus / Trolley
    # ------------------------------------------------------------------

    @cached_async(ttl=CACHE_TTL_TRANSIT_VIEW)
    async def get_transit_view(self, route: str) -> list[TransitViewBus]:
        """Active vehicles on a bus or trolley route."""
        resp = await self.client.get(
            TRANSIT_VIEW_URL, params={"route": route}
        )
        resp.raise_for_status()
        data = resp.json()

        # Response is {"bus": [...]} for a single route
        buses: list = []
        if isinstance(data, dict):
            buses = data.get("bus", [])
        elif isinstance(data, list):
            buses = data

        if not isinstance(buses, list):
            return []
        return _safe_parse(TransitViewBus, buses)

    # ------------------------------------------------------------------
    # Alerts / Detours
    # ------------------------------------------------------------------

    @cached_async(ttl=CACHE_TTL_ALERTS)
    async def get_alerts(
        self, route_id: str | None = None
    ) -> list[AlertData]:
        """Service alerts, optionally filtered by route_id."""
        params: dict[str, str] = {}
        if route_id:
            params["route_id"] = route_id
        resp = await self.client.get(ALERTS_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        return _safe_parse(AlertData, data)

    @cached_async(ttl=CACHE_TTL_ALERTS)
    async def get_bus_detours(
        self, route: str | None = None
    ) -> list[dict]:
        """Active bus detours, optionally for a specific route."""
        url = BUS_DETOURS_URL
        if route:
            url = f"{BUS_DETOURS_URL}{route}"
        resp = await self.client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []
