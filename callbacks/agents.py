"""
ReAct agents for travel booking system using LangChain callbacks.

Defines specialized agents:
  - Orchestrator: routes user requests to the correct specialist agent
  - CarBookingAgent: handles car rental booking
  - AirTicketAgent: handles flight booking
  - HotelReservationAgent: handles hotel reservation

All agents are LLM-driven via ReAct (Reasoning + Acting) using litellm + Ollama.
Prompt templates offer users selectable options when human-in-the-loop decisions are needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback event types
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


# ---------------------------------------------------------------------------
# Prompt templates for human-in-the-loop decisions
# ---------------------------------------------------------------------------

# Template for offering car type selection to the user
CAR_TYPE_SELECTION_PROMPT = """Please select a car type from the following options:

{options}

Enter the number of your choice (1-{count}), or type 'cancel' to abort:"""

# Template for offering flight selection
FLIGHT_SELECTION_PROMPT = """Please select a flight from the following options:

{options}

Enter the number of your choice (1-{count}), or type 'cancel' to abort:"""

# Template for offering hotel selection
HOTEL_SELECTION_PROMPT = """Please select a hotel from the following options:

{options}

Enter the number of your choice (1-{count}), or type 'cancel' to abort:"""

# Template for general user input
GENERAL_INPUT_PROMPT = """{message}

Please provide the required information:"""

# Template for confirmation
CONFIRMATION_PROMPT = """Please confirm the following:

{details}

Type 'yes' to confirm, 'no' to cancel, or provide alternative details:"""


def format_car_options(cars: list[dict[str, Any]]) -> str:
    """Format car options for the selection prompt."""
    lines = []
    for i, car in enumerate(cars, 1):
        lines.append(
            f"  {i}. {car.get('make', 'Unknown')} {car.get('model', 'Unknown')} "
            f"- ${car.get('price_per_day', '?')}/day"
        )
    return "\n".join(lines)


def format_flight_options(flights: list[dict[str, Any]]) -> str:
    """Format flight options for the selection prompt."""
    lines = []
    for i, flight in enumerate(flights, 1):
        lines.append(
            f"  {i}. {flight.get('airline', 'Unknown')} {flight.get('flight', '')} "
            f"- ${flight.get('price', '?')} "
            f"({flight.get('departure', '?')} - {flight.get('arrival', '?')})"
        )
    return "\n".join(lines)


def format_hotel_options(hotels: list[dict[str, Any]]) -> str:
    """Format hotel options for the selection prompt."""
    lines = []
    for i, hotel in enumerate(hotels, 1):
        lines.append(
            f"  {i}. {hotel.get('name', 'Unknown')} "
            f"({'★' * hotel.get('stars', 0)}) "
            f"- ${hotel.get('price_per_night', '?')}/night"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A registered tool an agent can invoke."""
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str] | None = None


# ----- Tool implementations (stubs that return structured strings) -----

def _search_cars(location: str, pickup_date: str, dropoff_date: str) -> str:
    """Search available cars."""
    cars = [
        {"id": "car_1", "make": "Toyota", "model": "Corolla", "price_per_day": 45},
        {"id": "car_2", "make": "Honda", "model": "Civic", "price_per_day": 50},
        {"id": "car_3", "make": "BMW", "model": "3 Series", "price_per_day": 90},
    ]
    return json.dumps({"available_cars": cars, "location": location})


def _book_car(car_id: str, location: str, pickup_date: str, dropoff_date: str) -> str:
    """Book a car."""
    return json.dumps({
        "status": "pending_confirmation",
        "car_id": car_id,
        "location": location,
        "pickup_date": pickup_date,
        "dropoff_date": dropoff_date,
        "message": f"Car {car_id} selected. Awaiting car type confirmation."
    })


def _select_car_type(car_id: str, car_type: str) -> str:
    """Confirm the specific car type/model after user selection."""
    return json.dumps({
        "status": "confirmed",
        "car_id": car_id,
        "car_type": car_type,
        "message": f"Car type '{car_type}' confirmed for {car_id}."
    })


def _search_flights(origin: str, destination: str, date: str) -> str:
    """Search available flights."""
    flights = [
        {"id": "fl_1", "airline": "BA", "flight": "BA123", "price": 350, "departure": "08:00", "arrival": "10:30"},
        {"id": "fl_2", "airline": "EasyJet", "flight": "EZY456", "price": 120, "departure": "14:00", "arrival": "16:15"},
        {"id": "fl_3", "airline": "Ryanair", "flight": "FR789", "price": 45, "departure": "06:30", "arrival": "08:45"},
    ]
    return json.dumps({"available_flights": flights, "origin": origin, "destination": destination})


def _book_air_ticket(flight_id: str, passenger_name: str, seat_class: str = "economy") -> str:
    """Book an air ticket."""
    return json.dumps({
        "status": "confirmed",
        "flight_id": flight_id,
        "passenger": passenger_name,
        "seat_class": seat_class,
        "message": f"Flight {flight_id} booked for {passenger_name} in {seat_class}."
    })


def _search_hotels(location: str, check_in: str, check_out: str) -> str:
    """Search available hotels."""
    hotels = [
        {"id": "ht_1", "name": "Grand Plaza", "stars": 5, "price_per_night": 250},
        {"id": "ht_2", "name": "City Inn", "stars": 3, "price_per_night": 90},
        {"id": "ht_3", "name": "Budget Stay", "stars": 2, "price_per_night": 50},
    ]
    return json.dumps({"available_hotels": hotels, "location": location})


def _book_hotel(hotel_id: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Book a hotel room."""
    return json.dumps({
        "status": "confirmed",
        "hotel_id": hotel_id,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "message": f"Hotel {hotel_id} booked for {guests} guest(s) from {check_in} to {check_out}."
    })


# ----- Tool registry -----

TOOL_REGISTRY: dict[str, Tool] = {
    "search_cars": Tool(
        name="search_cars",
        description="Search for available cars at a location between dates.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Pickup location"},
                "pickup_date": {"type": "string", "description": "Pickup date (YYYY-MM-DD)"},
                "dropoff_date": {"type": "string", "description": "Dropoff date (YYYY-MM-DD)"},
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
                "car_id": {"type": "string", "description": "Car ID to book"},
                "location": {"type": "string", "description": "Pickup location"},
                "pickup_date": {"type": "string", "description": "Pickup date (YYYY-MM-DD)"},
                "dropoff_date": {"type": "string", "description": "Dropoff date (YYYY-MM-DD)"},
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
                "car_id": {"type": "string", "description": "Car ID"},
                "car_type": {"type": "string", "description": "Car type/model chosen by user"},
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
                "origin": {"type": "string", "description": "Departure city/airport"},
                "destination": {"type": "string", "description": "Arrival city/airport"},
                "date": {"type": "string", "description": "Flight date (YYYY-MM-DD)"},
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
                "flight_id": {"type": "string", "description": "Flight ID to book"},
                "passenger_name": {"type": "string", "description": "Passenger full name"},
                "seat_class": {"type": "string", "description": "Seat class: economy, premium, business, first"},
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
                "location": {"type": "string", "description": "Hotel location/city"},
                "check_in": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                "check_out": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
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
                "hotel_id": {"type": "string", "description": "Hotel ID to book"},
                "check_in": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                "check_out": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                "guests": {"type": "integer", "description": "Number of guests"},
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
                "prompt": {"type": "string", "description": "The question or prompt to show the user"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of selectable options for the user (optional)",
                },
                "context": {
                    "type": "object",
                    "description": "Additional context data to pass through",
                },
            },
            "required": ["prompt"],
        },
        fn=None,  # handled by the callback system
    ),
}


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """A ReAct agent with a system prompt and a set of tools."""
    name: str
    description: str
    system_prompt: str
    tools: list[str]  # tool names from TOOL_REGISTRY


# ----- Orchestrator Agent -----

ORCHESTRATOR_SYSTEM_PROMPT = """You are a travel booking orchestrator. Your role is to:

    1. Receive a user's travel request.
    2. Determine which specialist agent(s) should handle it:
       - **CarBookingAgent** — for car rental bookings
       - **AirTicketAgent** — for flight / air ticket bookings
       - **HotelReservationAgent** — for hotel reservations
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

    When delegating, use the format:
    Action: request_human_input
    Action Input: {{"prompt": "I'll delegate this to the CarBookingAgent. What type of car are you looking for?", "context": {{"delegate_to": "CarBookingAgent", "original_request": "<user request>"}}}}"""

ORCHESTRATOR_AGENT = Agent(
    name="Orchestrator",
    description="Routes user requests to the correct specialist agent.",
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    tools=["request_human_input"],
)


# ----- Car Booking Agent -----

CAR_BOOKING_SYSTEM_PROMPT = """You are a car booking specialist agent. Your role is to:

    1. Search for available cars using `search_cars` when the user provides location and dates.
    2. Present the available options to the user using `request_human_input` with the options parameter.
    3. When the user selects a car, call `book_car` to initiate the booking.
    4. **Important**: After booking, use `request_human_input` to ask the user to confirm the car type.
       Include the available car types as options so the user can select.
    5. Once the user confirms the car type via their response, call `select_car_type` to finalise.
    6. Use `request_human_input` with a confirmation prompt when done.

    Always use the tools available to you. Never make up results.
    When presenting options to the user, always include them in the `options` parameter of `request_human_input`."""

CAR_BOOKING_AGENT = Agent(
    name="CarBookingAgent",
    description="Handles car rental search and booking.",
    system_prompt=CAR_BOOKING_SYSTEM_PROMPT,
    tools=["search_cars", "book_car", "select_car_type", "request_human_input"],
)


# ----- Air Ticket Agent -----

AIR_TICKET_SYSTEM_PROMPT = """You are an air ticket booking specialist agent. Your role is to:

    1. Search for available flights using `search_flights` when the user provides origin, destination, and date.
    2. Present the available options to the user using `request_human_input` with the options parameter.
    3. When the user selects a flight, call `book_air_ticket` to book it.
    4. Use `request_human_input` with a confirmation prompt when done.

    Always use the tools available to you. Never make up results.
    When presenting options to the user, always include them in the `options` parameter of `request_human_input`."""

AIR_TICKET_AGENT = Agent(
    name="AirTicketAgent",
    description="Handles flight search and booking.",
    system_prompt=AIR_TICKET_SYSTEM_PROMPT,
    tools=["search_flights", "book_air_ticket", "request_human_input"],
)


# ----- Hotel Reservation Agent -----

HOTEL_RESERVATION_SYSTEM_PROMPT = """You are a hotel reservation specialist agent. Your role is to:

    1. Search for available hotels using `search_hotels` when the user provides location and dates.
    2. Present the available options to the user using `request_human_input` with the options parameter.
    3. When the user selects a hotel, call `book_hotel` to book it.
    4. Use `request_human_input` with a confirmation prompt when done.

    Always use the tools available to you. Never make up results.
    When presenting options to the user, always include them in the `options` parameter of `request_human_input`."""

HOTEL_RESERVATION_AGENT = Agent(
    name="HotelReservationAgent",
    description="Handles hotel search and booking.",
    system_prompt=HOTEL_RESERVATION_SYSTEM_PROMPT,
    tools=["search_hotels", "book_hotel", "request_human_input"],
)


# ----- Agent registry -----

AGENT_REGISTRY: dict[str, Agent] = {
    ORCHESTRATOR_AGENT.name: ORCHESTRATOR_AGENT,
    CAR_BOOKING_AGENT.name: CAR_BOOKING_AGENT,
    AIR_TICKET_AGENT.name: AIR_TICKET_AGENT,
    HOTEL_RESERVATION_AGENT.name: HOTEL_RESERVATION_AGENT,
}


def get_agent(name: str) -> Agent | None:
    """Look up an agent by name."""
    return AGENT_REGISTRY.get(name)


def get_tool(name: str) -> Tool | None:
    """Look up a tool by name."""
    return TOOL_REGISTRY.get(name)