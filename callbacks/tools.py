"""
Tool implementations for the travel booking system.

All tools are proper LangChain ``StructuredTool`` instances (no hand-rolled
JSON parsing).  This module contains the implementation functions and the
``TOOL_REGISTRY`` that maps tool names to their ``StructuredTool`` objects.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock tool implementations
# Each returns a plain Python dict (the framework handles serialisation).
# ---------------------------------------------------------------------------

def search_cars(
    location: str,
    pickup_date: str,
    dropoff_date: str,
) -> dict[str, Any]:
    """Return mock available cars for *location* between *pickup_date* and *dropoff_date*."""
    return {
        "available_cars": [
            {"id": "car_1", "make": "Toyota", "model": "Corolla", "price_per_day": 45},
            {"id": "car_2", "make": "Honda", "model": "Civic", "price_per_day": 50},
            {"id": "car_3", "make": "BMW", "model": "3 Series", "price_per_day": 90},
        ],
        "location": location,
    }


def book_car(
    car_id: str,
    location: str,
    pickup_date: str,
    dropoff_date: str,
) -> dict[str, Any]:
    """Book car *car_id* and return pending‑confirmation status."""
    return {
        "status": "pending_confirmation",
        "car_id": car_id,
        "location": location,
        "pickup_date": pickup_date,
        "dropoff_date": dropoff_date,
    }


def select_car_type(car_id: str, car_type: str) -> dict[str, Any]:
    """Confirm the specific car type/model for *car_id*."""
    return {
        "status": "confirmed",
        "car_id": car_id,
        "car_type": car_type,
    }


def search_flights(
    origin: str,
    destination: str,
    date: str,
) -> dict[str, Any]:
    """Search for available flights between cities on *date*."""
    return {
        "available_flights": [
            {"id": "fl_1", "airline": "BA", "flight": "BA123", "price": 350,
             "departure": "08:00", "arrival": "10:30"},
            {"id": "fl_2", "airline": "EasyJet", "flight": "EZY456", "price": 120,
             "departure": "14:00", "arrival": "16:15"},
            {"id": "fl_3", "airline": "Ryanair", "flight": "FR789", "price": 45,
             "departure": "06:30", "arrival": "08:45"},
        ],
        "origin": origin,
        "destination": destination,
    }


def book_air_ticket(
    flight_id: str,
    passenger_name: str,
    seat_class: str = "economy",
) -> dict[str, Any]:
    """Book flight *flight_id* for *passenger_name*."""
    return {
        "status": "confirmed",
        "flight_id": flight_id,
        "passenger": passenger_name,
        "seat_class": seat_class,
    }


def search_hotels(
    location: str,
    check_in: str,
    check_out: str,
) -> dict[str, Any]:
    """Search for available hotels at *location* between dates."""
    return {
        "available_hotels": [
            {"id": "ht_1", "name": "Grand Plaza", "stars": 5, "price_per_night": 250},
            {"id": "ht_2", "name": "City Inn", "stars": 3, "price_per_night": 90},
            {"id": "ht_3", "name": "Budget Stay", "stars": 2, "price_per_night": 50},
        ],
        "location": location,
    }


def book_hotel(
    hotel_id: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
) -> dict[str, Any]:
    """Book hotel *hotel_id*."""
    return {
        "status": "confirmed",
        "hotel_id": hotel_id,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
    }


def delegate_to_agent(agent_name: str, request: str) -> dict[str, Any]:
    """Delegate a user request to a specialist agent.

    Args:
        agent_name: The name of the agent to delegate to.
            One of: CarBookingAgent, AirTicketAgent, HotelReservationAgent.
        request: The user's original request to pass to the specialist agent.

    Returns:
        A delegation response that the processor interprets to start
        the specialist agent's ReAct loop.
    """
    return {
        "status": "delegating",
        "delegate_to": agent_name,
        "original_request": request,
    }


# ---------------------------------------------------------------------------
# LangChain StructuredTool definitions
# ---------------------------------------------------------------------------

search_cars_tool = StructuredTool.from_function(
    func=search_cars,
    name="search_cars",
    description="Search for available cars at a location between dates.",
)

book_car_tool = StructuredTool.from_function(
    func=book_car,
    name="book_car",
    description="Book a car. Returns pending confirmation — user must confirm car type.",
)

select_car_type_tool = StructuredTool.from_function(
    func=select_car_type,
    name="select_car_type",
    description="Confirm the specific car type/model after user selects one.",
)

search_flights_tool = StructuredTool.from_function(
    func=search_flights,
    name="search_flights",
    description="Search for available flights between cities on a date.",
)

book_air_ticket_tool = StructuredTool.from_function(
    func=book_air_ticket,
    name="book_air_ticket",
    description="Book an airline ticket for a passenger.",
)

search_hotels_tool = StructuredTool.from_function(
    func=search_hotels,
    name="search_hotels",
    description="Search for available hotels at a location between dates.",
)

book_hotel_tool = StructuredTool.from_function(
    func=book_hotel,
    name="book_hotel",
    description="Book a hotel room.",
)

delegate_to_agent_tool = StructuredTool.from_function(
    func=delegate_to_agent,
    name="delegate_to_agent",
    description=(
        "Delegate a user request to a specialist agent. "
        "Use this to hand off a task to one of: "
        "CarBookingAgent, AirTicketAgent, HotelReservationAgent. "
        "Provide the agent_name and the user's original request."
    ),
)

# ---------------------------------------------------------------------------
# Tool registry and lookup
# ---------------------------------------------------------------------------
# Note: ``request_human_input`` is NOT a LangChain tool — it is handled
# directly by the processor callback system to pause the ReAct loop.
# It still needs a definition so agents can reference it.

TOOL_REGISTRY: dict[str, StructuredTool] = {
    "search_cars": search_cars_tool,
    "book_car": book_car_tool,
    "select_car_type": select_car_type_tool,
    "search_flights": search_flights_tool,
    "book_air_ticket": book_air_ticket_tool,
    "search_hotels": search_hotels_tool,
    "book_hotel": book_hotel_tool,
    "delegate_to_agent": delegate_to_agent_tool,
}


def get_tool(name: str) -> StructuredTool | None:
    """Look up a LangChain tool by name.

    Returns ``None`` for ``request_human_input`` — that tool is handled
    externally by the processor callback system.
    """
    return TOOL_REGISTRY.get(name)