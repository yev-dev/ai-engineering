#!/usr/bin/env python3
"""
CLI for the travel booking agent system using LangChain callbacks.

Provides an interactive command-line interface where users can:
  - Make travel requests in natural language
  - Respond to agent prompts (human-in-the-loop) with selectable options
  - View booking status

Uses LangChain's callback system for human-in-the-loop interaction
with litellm + Ollama (gemma4:e4b).

Usage:
    python cli.py
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents import CallbackEvent, CallbackEventType
from processor import TravelBookingProcessor

logger = logging.getLogger("cli")

# ---------------------------------------------------------------------------
# Prompt templates (presentation layer)
# ---------------------------------------------------------------------------

CAR_TYPE_SELECTION_PROMPT = """Please select a car type from the following options:

{options}

Enter the number of your choice (1-{count}), or type 'cancel' to abort:"""

FLIGHT_SELECTION_PROMPT = """Please select a flight from the following options:

{options}

Enter the number of your choice (1-{count}), or type 'cancel' to abort:"""

HOTEL_SELECTION_PROMPT = """Please select a hotel from the following options:

{options}

Enter the number of your choice (1-{count}), or type 'cancel' to abort:"""

GENERAL_INPUT_PROMPT = """{message}

Please provide the required information:"""

CONFIRMATION_PROMPT = """Please confirm the following:

{details}

Type 'yes' to confirm, 'no' to cancel, or provide alternative details:"""


def format_car_options(cars: list[dict[str, Any]]) -> str:
    lines = []
    for i, car in enumerate(cars, 1):
        lines.append(
            f"  {i}. {car.get('make', 'Unknown')} {car.get('model', 'Unknown')} "
            f"- ${car.get('price_per_day', '?')}/day"
        )
    return "\n".join(lines)


def format_flight_options(flights: list[dict[str, Any]]) -> str:
    lines = []
    for i, flight in enumerate(flights, 1):
        lines.append(
            f"  {i}. {flight.get('airline', 'Unknown')} {flight.get('flight', '')} "
            f"- ${flight.get('price', '?')} "
            f"({flight.get('departure', '?')} - {flight.get('arrival', '?')})"
        )
    return "\n".join(lines)


def format_hotel_options(hotels: list[dict[str, Any]]) -> str:
    lines = []
    for i, hotel in enumerate(hotels, 1):
        lines.append(
            f"  {i}. {hotel.get('name', 'Unknown')} "
            f"({'★' * hotel.get('stars', 0)}) "
            f"- ${hotel.get('price_per_night', '?')}/night"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Event display
# ---------------------------------------------------------------------------

def _event_display(event: CallbackEvent) -> str | None:
    """Return a display string for an event, or None if it should be silent."""
    p = event.payload
    tag = f"[{p.get('agent', 'System')}]"

    if event.type == CallbackEventType.AWAITING_USER_INPUT:
        text = f"\n{tag} {p.get('prompt', '')}"
        options = p.get("options")
        if options:
            items = "\n".join(f"  {i}. {o}" for i, o in enumerate(options, 1))
            text += f"\n\n{items}\n\n  (Enter the number of your choice, or type your response)"
        return text

    if event.type == CallbackEventType.AGENT_COMPLETED:
        answer = (p.get("result", {}) or {}).get("final_answer", "")
        if "Final Answer:" in answer:
            answer = answer.split("Final Answer:")[-1].strip()
        return f"\n{tag} {answer}" if answer else f"\n{tag} Task completed."

    if event.type == CallbackEventType.BOOKING_CONFIRMED:
        return f"\n[System] ✅ Booking confirmed!\n{json.dumps(p.get('details', {}), indent=2)}"

    if event.type == CallbackEventType.BOOKING_FAILED:
        return f"\n[System] ❌ Booking failed: {p.get('reason', 'Unknown error')}"

    if event.type == CallbackEventType.ERROR:
        return f"\n[System] ⚠️  Error: {p.get('message', 'Unknown error')}"

    if event.type == CallbackEventType.CAR_TYPE_CONFIRMED:
        return f"\n[System] Car type '{p.get('car_type', '')}' confirmed for {p.get('car_id', '')}."

    if event.type == CallbackEventType.SELECT_CAR:
        lines = [f"\n[CarBookingAgent] Car {p.get('car_id', 'unknown')} selected."]
        cars = p.get("available_cars", [])
        if cars:
            lines.append("Available car options:")
            lines.append(format_car_options(cars))
        prompt = CAR_TYPE_SELECTION_PROMPT.format(
            options=format_car_options(cars),
            count=len(cars),
        ) if cars else ""
        if prompt:
            lines.append(f"\n{prompt}")
        return "\n".join(lines)

    return None


# ---------------------------------------------------------------------------
# CLI Application
# ---------------------------------------------------------------------------

class TravelBookingCLI:
    """Interactive CLI for the travel booking system."""

    def __init__(self) -> None:
        self.processor = TravelBookingProcessor()
        self.running = True
        self.waiting_for_input = False

    def _print_header(self) -> None:
        print("=" * 60)
        print("  Travel Booking Agent System")
        print("  Powered by ReAct Agents + LangChain Callbacks")
        print("  Ollama (gemma4:e4b) via litellm")
        print("=" * 60)
        print()
        print("You can ask me to book cars, flights, or hotels.")
        print("Type 'quit' to exit, 'help' for commands.")
        print()

    def _print_help(self) -> None:
        print()
        print("Commands:")
        print("  help              - Show this help message")
        print("  quit / exit       - Exit the application")
        print("  reset             - Reset the conversation")
        print()
        print("Example requests:")
        print('  "Book a car in London for next week"')
        print('  "Find flights from London to Paris"')
        print('  "Book a hotel in New York"')
        print('  "I need a car, flight, and hotel for my trip to Barcelona"')
        print()

    def _drain_events(self, events: list[CallbackEvent]) -> None:
        """Display all events from a list + any queued in the handler."""
        for event in events:
            text = _event_display(event)
            if text:
                print(text)

        # Drain the handler's event queue
        while self.processor.handler.has_events():
            event = self.processor.handler.poll()
            if event:
                text = _event_display(event)
                if text:
                    print(text)

    def handle_user_input(self, user_input: str) -> None:
        """Process user input and display resulting events."""
        if self.waiting_for_input:
            events = self.processor.process_user_response(user_input)
            self.waiting_for_input = False
        else:
            events = self.processor.process_user_request(user_input)

        self._drain_events(events)

        # If the handler is now waiting, update the flag
        if self.processor.handler.waiting_for_input:
            self.waiting_for_input = True

    def run(self) -> None:
        """Run the CLI main loop."""
        self._print_header()

        while self.running:
            try:
                prompt = "> " if self.waiting_for_input else "\nYou: "
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                cmd = user_input.lower()
                if cmd in ("quit", "exit"):
                    print("\nThank you for using the Travel Booking Agent System. Goodbye!")
                    self.running = False
                    break

                if cmd == "help":
                    self._print_help()
                    continue

                if cmd == "reset":
                    self.processor.reset()
                    self.waiting_for_input = False
                    print("\n[System] Conversation reset.")
                    continue

                self.handle_user_input(user_input)

            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                self.running = False
                break
            except Exception as e:
                logger.exception("Unexpected error")
                print(f"\n[System] ⚠️  Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cli = TravelBookingCLI()
    cli.run()


if __name__ == "__main__":
    main()
