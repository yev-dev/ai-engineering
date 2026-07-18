"""
Agent and tool definitions for the travel booking system.

Pure configuration layer — defines what agents exist, what tools they
can use, and the tool implementations. No presentation logic, no event
management, no loop orchestration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class CallbackEventType(str, Enum):
    """Events that flow through the callback system."""
    USER_REQUEST = "user_request"
    AWAITING_USER_INPUT = "awaiting_user_input"
    SELECT_CAR = "select_car"
    CAR_TYPE_CONFIRMED = "car_type_confirmed"
    BOOKING_CONFIRMED = "booking_confirmed"
    BOOKING_FAILED = "booking_failed"
    AGENT_COMPLETED = "agent_completed"
    ERROR = "error"


@dataclass
class CallbackEvent:
    """A message / event passed between agents and the handler via callbacks."""
    type: CallbackEventType
    payload: dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"

@dataclass
class Tool:
    """A registered tool an agent can invoke."""
    name: str
    description: str
    parameters: dict[str, Any]          # JSON Schema for the LLM
    fn: Callable[..., str] | None = None  # implementation, None = handled externally


@dataclass
class Agent:
    """A ReAct agent with a system prompt and set of tool names."""
    name: str
    description: str
    system_prompt: str
    tools: list[str]  # keys into TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Tool implementations (stubs returning structured JSON)
# ---------------------------------------------------------------------------

def _search_cars(location: str, pickup_date: str, dropoff_date: str) -> str:
    return json.dumps({
        "available_cars": [
            {"id": "car_1", "make": "Toyota", "model": "Corolla", "price_per_day": 45},
            {"id": "car_2", "make": "Honda", "model": "Civic", "price_per_day": 50},
            {"id": "car_3", "make": "BMW", "model": "3 Series", "price_per_day": 90},
        ],
        "location": location,
    })


def _book_car(car_id: str, location: str, pickup_date: str, dropoff_date: str) -> str:
    return json.dumps({
        "status": "pending_confirmation",
        "car_id": car_id,
        "location": location,
        "pickup_date": pickup_date,
        "dropoff_date": dropoff_date,
    })


def _select_car_type(car_id: str, car_type: str) -> str:
    return json.dumps({
        "status": "confirmed",
        "car_id": car_id,
        "car_type": car_type,
    })


def _search_flights(origin: str, destination: str, date: str) -> str:
    return json.dumps({
        "available_flights": [
            {"id": "fl_1", "airline": "BA", "flight": "BA123", "price": 350, "departure": "08:00", "arrival": "10:30"},
            {"id": "fl_2", "airline": "EasyJet", "flight": "EZY456", "price": 120, "departure": "14:00", "arrival": "16:15"},
            {"id": "fl_3", "airline": "Ryanair", "flight": "FR789", "price": 45, "departure": "06:30", "arrival": "08:45"},
        ],
        "origin": origin,
        "destination": destination,
    })


def _book_air_ticket(flight_id: str, passenger_name: str, seat_class: str = "economy") -> str:
    return json.dumps({
        "status": "confirmed",
        "flight_id": flight_id,
        "passenger": passenger_name,
        "seat_class": seat_class,
    })


def _search_hotels(location: str, check_in: str, check_out: str) -> str:
    return json.dumps({
        "available_hotels": [
            {"id": "ht_1", "name": "Grand Plaza", "stars": 5, "price_per_night": 250},
            {"id": "ht_2", "name": "City Inn", "stars": 3, "price_per_night": 90},
            {"id": "ht_3", "name": "Budget Stay", "stars": 2, "price_per_night": 50},
        ],
        "location": location,
    })


def _book_hotel(hotel_id: str, check_in: str, check_out: str, guests: int = 1) -> str:
    return json.dumps({
        "status": "confirmed",
        "hotel_id": hotel_id,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
    })


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Tool] = {
    "search_cars": Tool(
        name="search_cars",
        description="Search for available cars at a location between dates.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "pickup_date": {"type": "string"},
                "dropoff_date": {"type": "string"},
            },
            "required": ["location", "pickup_date", "dropoff_date"],
        },
        fn=_search_cars,
    ),
    "book_car": Tool(
        name="book_car",
        description="Book a car. Returns pending confirmation — user must confirm car type.",
        parameters={
            "type": "object",
            "properties": {
                "car_id": {"type": "string"},
                "location": {"type": "string"},
                "pickup_date": {"type": "string"},
                "dropoff_date": {"type": "string"},
            },
            "required": ["car_id", "location", "pickup_date", "dropoff_date"],
        },
        fn=_book_car,
    ),
    "select_car_type": Tool(
        name="select_car_type",
        description="Confirm the specific car type/model after user selects one.",
        parameters={
            "type": "object",
            "properties": {
                "car_id": {"type": "string"},
                "car_type": {"type": "string"},
            },
            "required": ["car_id", "car_type"],
        },
        fn=_select_car_type,
    ),
    "search_flights": Tool(
        name="search_flights",
        description="Search for available flights between cities on a date.",
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["origin", "destination", "date"],
        },
        fn=_search_flights,
    ),
    "book_air_ticket": Tool(
        name="book_air_ticket",
        description="Book an airline ticket for a passenger.",
        parameters={
            "type": "object",
            "properties": {
                "flight_id": {"type": "string"},
                "passenger_name": {"type": "string"},
                "seat_class": {"type": "string", "enum": ["economy", "premium", "business", "first"]},
            },
            "required": ["flight_id", "passenger_name"],
        },
        fn=_book_air_ticket,
    ),
    "search_hotels": Tool(
        name="search_hotels",
        description="Search for available hotels at a location between dates.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "check_in": {"type": "string"},
                "check_out": {"type": "string"},
            },
            "required": ["location", "check_in", "check_out"],
        },
        fn=_search_hotels,
    ),
    "book_hotel": Tool(
        name="book_hotel",
        description="Book a hotel room.",
        parameters={
            "type": "object",
            "properties": {
                "hotel_id": {"type": "string"},
                "check_in": {"type": "string"},
                "check_out": {"type": "string"},
                "guests": {"type": "integer"},
            },
            "required": ["hotel_id", "check_in", "check_out"],
        },
        fn=_book_hotel,
    ),
    "request_human_input": Tool(
        name="request_human_input",
        description="Request input from the human user. Use this when you need the user to make a decision or provide information.",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "context": {"type": "object"},
            },
            "required": ["prompt"],
        },
        fn=None,  # handled by the callback system
    ),
}


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """You are a travel booking orchestrator. Your role is to:

1. Receive a user's travel request.
2. Determine which specialist agent(s) should handle it:
   - CarBookingAgent — for car rental bookings
   - AirTicketAgent — for flight / air ticket bookings
   - HotelReservationAgent — for hotel reservations
3. Delegate by emitting an event with the appropriate agent name and the relevant details.
4. If the request involves multiple services (e.g. a full trip), delegate to each agent in sequence.

Rules:
- NEVER try to book anything yourself. Always delegate to the correct specialist agent.
- If the user's intent is unclear, ask clarifying questions by requesting human input.
- Use the `request_human_input` tool to ask the user for clarification or decisions.

Available specialist agents:
- CarBookingAgent: handles car rental search and booking
- AirTicketAgent: handles flight search and booking
- HotelReservationAgent: handles hotel search and booking

When delegating, use:
Action: request_human_input
Action Input: {{"prompt": "I'll delegate this to the <agent>. What do you need?", "context": {{"delegate_to": "<AgentName>", "original_request": "<user request>"}}}}"""

CAR_BOOKING_SYSTEM_PROMPT = """You are a car booking specialist agent. Your role is to:

1. Search for available cars using `search_cars` when the user provides location and dates.
2. Present the available options to the user using `request_human_input` with the options parameter.
3. When the user selects a car, call `book_car` to initiate the booking.
4. After booking, use `request_human_input` to ask the user to confirm the car type.
5. Once the user confirms the car type, call `select_car_type` to finalise.
6. Use `request_human_input` with a confirmation prompt when done.

Always use the tools available to you. Never make up results."""

AIR_TICKET_SYSTEM_PROMPT = """You are an air ticket booking specialist agent. Your role is to:

1. Search for available flights using `search_flights` when the user provides origin, destination, and date.
2. Present the available options to the user using `request_human_input` with the options parameter.
3. When the user selects a flight, call `book_air_ticket` to book it.
4. Use `request_human_input` with a confirmation prompt when done.

Always use the tools available to you. Never make up results."""

HOTEL_RESERVATION_SYSTEM_PROMPT = """You are a hotel reservation specialist agent. Your role is to:

1. Search for available hotels using `search_hotels` when the user provides location and dates.
2. Present the available options to the user using `request_human_input` with the options parameter.
3. When the user selects a hotel, call `book_hotel` to book it.
4. Use `request_human_input` with a confirmation prompt when done.

Always use the tools available to you. Never make up results."""

AGENT_REGISTRY: dict[str, Agent] = {
    "Orchestrator": Agent(
        name="Orchestrator",
        description="Routes user requests to the correct specialist agent.",
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=["request_human_input"],
    ),
    "CarBookingAgent": Agent(
        name="CarBookingAgent",
        description="Handles car rental search and booking.",
        system_prompt=CAR_BOOKING_SYSTEM_PROMPT,
        tools=["search_cars", "book_car", "select_car_type", "request_human_input"],
    ),
    "AirTicketAgent": Agent(
        name="AirTicketAgent",
        description="Handles flight search and booking.",
        system_prompt=AIR_TICKET_SYSTEM_PROMPT,
        tools=["search_flights", "book_air_ticket", "request_human_input"],
    ),
    "HotelReservationAgent": Agent(
        name="HotelReservationAgent",
        description="Handles hotel search and booking.",
        system_prompt=HOTEL_RESERVATION_SYSTEM_PROMPT,
        tools=["search_hotels", "book_hotel", "request_human_input"],
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_agent(name: str) -> Agent | None:
    return AGENT_REGISTRY.get(name)


def get_tool(name: str) -> Tool | None:
    return TOOL_REGISTRY.get(name)