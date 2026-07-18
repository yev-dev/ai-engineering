"""
Event handler for the travel booking agent system.

Handles human-in-the-loop events:
  - AWAITING_USER_INPUT: prompts the user for input
  - SELECT_CAR: asks user to confirm car type
  - CAR_TYPE_CONFIRMED: processes car type confirmation
  - AGENT_COMPLETED: signals agent task completion
  - ERROR: handles errors

Uses litellm + Ollama (gemma4:e4b) for LLM-driven processing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from agents import (
    AGENT_REGISTRY,
    TOOL_REGISTRY,
    Event,
    EventType,
    Agent,
    Tool,
    get_agent,
    get_tool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event handler registry
# ---------------------------------------------------------------------------

EventHandler = Callable[[Event], list[Event] | None]


class EventHandlerRegistry:
    """Registry mapping EventType -> list of handlers."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}

    def register(self, event_type: EventType) -> Callable[[EventHandler], EventHandler]:
        """Decorator to register a handler for an event type."""
        def decorator(handler: EventHandler) -> EventHandler:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            return handler
        return decorator

    def dispatch(self, event: Event) -> list[Event]:
        """Dispatch an event to all registered handlers. Returns emitted events."""
        results: list[Event] = []
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            emitted = handler(event)
            if emitted:
                results.extend(emitted)
        return results

    def get_handlers(self, event_type: EventType) -> list[EventHandler]:
        """Get all handlers for an event type."""
        return self._handlers.get(event_type, [])


# Global registry
registry = EventHandlerRegistry()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@registry.register(EventType.AWAITING_USER_INPUT)
def handle_awaiting_user_input(event: Event) -> list[Event]:
    """Handle an event that requires user input.

    The payload should contain:
      - "prompt": the question/message to show the user
      - "agent": the agent name that is waiting (optional)
      - "context": any context data (optional)

    This handler prints the prompt and returns an event that the CLI
    layer will pick up to read user input.
    """
    prompt = event.payload.get("prompt", "Please provide input:")
    agent_name = event.payload.get("agent", "System")
    context = event.payload.get("context", {})

    print(f"\n[{agent_name}] {prompt}")

    # Return a response event that the CLI will handle
    return [
        Event(
            type=EventType.AWAITING_USER_INPUT,
            payload={
                "prompt": prompt,
                "agent": agent_name,
                "context": context,
                "session_id": event.session_id,
            },
            session_id=event.session_id,
        )
    ]


@registry.register(EventType.SELECT_CAR)
def handle_select_car(event: Event) -> list[Event]:
    """Handle a car selection event.

    The payload should contain:
      - "car_id": the car that was selected
      - "location": pickup location
      - "pickup_date": pickup date
      - "dropoff_date": dropoff date
      - "available_cars": list of available car options (optional)

    This asks the user to confirm the car type.
    """
    car_id = event.payload.get("car_id", "unknown")
    location = event.payload.get("location", "")
    pickup_date = event.payload.get("pickup_date", "")
    dropoff_date = event.payload.get("dropoff_date", "")
    available_cars = event.payload.get("available_cars", [])

    print(f"\n[CarBookingAgent] Car {car_id} has been selected.")
    if available_cars:
        print("Available car options:")
        for car in available_cars:
            print(f"  - {car.get('id')}: {car.get('make')} {car.get('model')} "
                  f"(${car.get('price_per_day')}/day)")
    print(f"Location: {location}, Pickup: {pickup_date}, Dropoff: {dropoff_date}")

    return [
        Event(
            type=EventType.SELECT_CAR,
            payload={
                "car_id": car_id,
                "location": location,
                "pickup_date": pickup_date,
                "dropoff_date": dropoff_date,
                "available_cars": available_cars,
                "session_id": event.session_id,
            },
            session_id=event.session_id,
        )
    ]


@registry.register(EventType.CAR_TYPE_CONFIRMED)
def handle_car_type_confirmed(event: Event) -> list[Event]:
    """Handle a car type confirmation from the user.

    The payload should contain:
      - "car_id": the car ID
      - "car_type": the car type/model confirmed by the user
    """
    car_id = event.payload.get("car_id", "unknown")
    car_type = event.payload.get("car_type", "")

    logger.info("Car type confirmed: car_id=%s, car_type=%s", car_id, car_type)
    print(f"\n[CarBookingAgent] Car type '{car_type}' confirmed for {car_id}.")

    return [
        Event(
            type=EventType.CAR_TYPE_CONFIRMED,
            payload={
                "car_id": car_id,
                "car_type": car_type,
                "session_id": event.session_id,
            },
            session_id=event.session_id,
        )
    ]


@registry.register(EventType.AGENT_COMPLETED)
def handle_agent_completed(event: Event) -> list[Event]:
    """Handle an agent completion event.

    The payload should contain:
      - "agent": the agent name that completed
      - "result": the result data (optional)
    """
    agent_name = event.payload.get("agent", "Unknown")
    result = event.payload.get("result", {})

    logger.info("Agent '%s' completed with result: %s", agent_name, result)
    print(f"\n[System] Agent '{agent_name}' has completed its task.")

    return [
        Event(
            type=EventType.AGENT_COMPLETED,
            payload={
                "agent": agent_name,
                "result": result,
                "session_id": event.session_id,
            },
            session_id=event.session_id,
        )
    ]


@registry.register(EventType.BOOKING_CONFIRMED)
def handle_booking_confirmed(event: Event) -> list[Event]:
    """Handle a booking confirmation event."""
    details = event.payload.get("details", {})
    logger.info("Booking confirmed: %s", details)
    print(f"\n[System] Booking confirmed: {json.dumps(details, indent=2)}")
    return None


@registry.register(EventType.BOOKING_FAILED)
def handle_booking_failed(event: Event) -> list[Event]:
    """Handle a booking failure event."""
    reason = event.payload.get("reason", "Unknown error")
    logger.error("Booking failed: %s", reason)
    print(f"\n[System] Booking failed: {reason}")
    return None


@registry.register(EventType.ERROR)
def handle_error(event: Event) -> list[Event]:
    """Handle an error event."""
    error_msg = event.payload.get("message", "Unknown error")
    agent_name = event.payload.get("agent", "System")
    logger.error("Error from '%s': %s", agent_name, error_msg)
    print(f"\n[Error - {agent_name}] {error_msg}")
    return None


# ---------------------------------------------------------------------------
# Main event processing function
# ---------------------------------------------------------------------------

def process_event(event: Event) -> list[Event]:
    """Process a single event through the handler registry.

    Args:
        event: The event to process.

    Returns:
        A list of events emitted by handlers (may be empty).
    """
    logger.debug("Processing event: %s", event)
    return registry.dispatch(event)


def process_events(events: list[Event]) -> list[Event]:
    """Process a batch of events.

    Args:
        events: List of events to process.

    Returns:
        All events emitted during processing.
    """
    all_emitted: list[Event] = []
    for event in events:
        emitted = process_event(event)
        if emitted:
            all_emitted.extend(emitted)
    return all_emitted