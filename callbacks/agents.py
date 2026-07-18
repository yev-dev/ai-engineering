"""
Agent definitions for the travel booking system.

Pure configuration layer — defines what agents exist, what tools they
can use, and the lookup helpers.  Tool implementations live in ``tools.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types (used across the callback event system)
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
class Agent:
    """A ReAct agent with a system prompt and set of tool names.

    ``tools`` is a list of keys into ``TOOL_REGISTRY`` (defined in ``tools.py``).
    """
    name: str
    description: str
    system_prompt: str
    tools: list[str]


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """You are a travel booking orchestrator. Your role is to:

1. Receive a user's travel request.
2. Determine which specialist agent(s) should handle it:
   - CarBookingAgent — for car rental bookings
   - AirTicketAgent — for flight / air ticket bookings
   - HotelReservationAgent — for hotel reservations
3. Delegate by calling the `delegate_to_agent` tool with the correct agent name and the original request.
4. If the request involves multiple services (e.g. a full trip), delegate to each agent in sequence.

Rules:
- NEVER try to book anything yourself. Always delegate to the correct specialist agent.
- If the user's intent is unclear, ask clarifying questions by requesting human input with `request_human_input`.
- When you have a clear intent, ALWAYS use `delegate_to_agent` to hand off to the specialist.
- Do NOT use `request_human_input` to delegate. Only use it for asking clarifying questions.

Available specialist agents:
- CarBookingAgent: handles car rental search and booking
- AirTicketAgent: handles flight search and booking
- HotelReservationAgent: handles hotel search and booking

When delegating, use:
Action: delegate_to_agent
Action Input: {{"agent_name": "AgentName", "request": "the user's original request"}}"""

CAR_BOOKING_SYSTEM_PROMPT = """You are a car booking specialist agent. Your role is to:

1. Search for available cars using `search_cars` when the user provides location and dates.
2. Present the available options to the user using `request_human_input` with the options parameter.
3. When the user selects a car, call `book_car` to initiate the booking.
4. After booking, use `request_human_input` to ask the user to confirm the car type.
5. Once the user confirms the car type, call `select_car_type` to finalise.
6. Use `request_human_input` with a confirmation prompt when done.

IMPORTANT: You must use your domain tools (`search_cars`, `book_car`, `select_car_type`)
BEFORE asking the user for input with `request_human_input`. Do not ask the user for
information you already have. If the user provides location and dates, call `search_cars`
immediately — do not ask for confirmation first.

Always use the tools available to you. Never make up results."""

AIR_TICKET_SYSTEM_PROMPT = """You are an air ticket booking specialist agent. Your role is to:

1. Search for available flights using `search_flights` when the user provides origin, destination, and date.
2. Present the available options to the user using `request_human_input` with the options parameter.
3. When the user selects a flight, call `book_air_ticket` to book it.
4. Use `request_human_input` with a confirmation prompt when done.

IMPORTANT: You must use your domain tools (`search_flights`, `book_air_ticket`)
BEFORE asking the user for input with `request_human_input`. Do not ask the user for
information you already have. If the user provides origin, destination, and date,
call `search_flights` immediately — do not ask for confirmation first.

Always use the tools available to you. Never make up results."""

HOTEL_RESERVATION_SYSTEM_PROMPT = """You are a hotel reservation specialist agent. Your role is to:

1. Search for available hotels using `search_hotels` when the user provides location and dates.
2. Present the available options to the user using `request_human_input` with the options parameter.
3. When the user selects a hotel, call `book_hotel` to book it.
4. Use `request_human_input` with a confirmation prompt when done.

IMPORTANT: You must use your domain tools (`search_hotels`, `book_hotel`)
BEFORE asking the user for input with `request_human_input`. Do not ask the user for
information you already have. If the user provides location and dates, call `search_hotels`
immediately — do not ask for confirmation first.

Always use the tools available to you. Never make up results."""

AGENT_REGISTRY: dict[str, Agent] = {
    "Orchestrator": Agent(
        name="Orchestrator",
        description="Routes user requests to the correct specialist agent.",
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=["delegate_to_agent", "request_human_input"],
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
    """Look up an agent by name."""
    return AGENT_REGISTRY.get(name)