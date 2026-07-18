#!/usr/bin/env python3
"""
CLI for the travel booking agent system using LangChain callbacks.

Provides an interactive command-line interface where users can:
  - Make travel requests in natural language
  - Respond to agent prompts (human-in-the-loop) with selectable options
  - Confirm car types, flights, and hotels
  - View booking status

Uses LangChain's callback system for human-in-the-loop interaction
with litellm + Ollama (gemma4:e4b).

Usage:
    python cli.py
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from agents import (
    CallbackEvent,
    CallbackEventType,
    CAR_TYPE_SELECTION_PROMPT,
    FLIGHT_SELECTION_PROMPT,
    HOTEL_SELECTION_PROMPT,
    GENERAL_INPUT_PROMPT,
    CONFIRMATION_PROMPT,
    format_car_options,
    format_flight_options,
    format_hotel_options,
)
from handler import process_callback_event
from processor import TravelBookingProcessor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


# ---------------------------------------------------------------------------
# CLI Application
# ---------------------------------------------------------------------------

class TravelBookingCLI:
    """Interactive CLI for the travel booking system using LangChain callbacks."""

    def __init__(self) -> None:
        self.processor = TravelBookingProcessor()
        self.running = True
        self.waiting_for_input = False

    def print_header(self) -> None:
        """Print the application header."""
        print("=" * 60)
        print("  Travel Booking Agent System")
        print("  Powered by ReAct Agents + LangChain Callbacks")
        print("  Ollama (gemma4:e4b) via litellm")
        print("=" * 60)
        print()
        print("You can ask me to book cars, flights, or hotels.")
        print("Type 'quit' to exit, 'help' for commands.")
        print()

    def print_help(self) -> None:
        """Print help information."""
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

    def _format_options_for_display(self, options: list[str] | None) -> str:
        """Format a list of options for display to the user."""
        if not options:
            return ""
        lines = []
        for i, option in enumerate(options, 1):
            lines.append(f"  {i}. {option}")
        return "\n".join(lines)

    def _handle_awaiting_input_event(self, event: CallbackEvent) -> None:
        """Handle an AWAITING_USER_INPUT event by displaying the prompt."""
        payload = event.payload
        prompt = payload.get("prompt", "Please provide input:")
        options = payload.get("options", None)
        agent_name = payload.get("agent", "System")

        print(f"\n[{agent_name}] {prompt}")

        if options:
            print()
            print(self._format_options_for_display(options))
            print()
            print("  (Enter the number of your choice, or type your response)")

        self.waiting_for_input = True

    def _handle_select_car_event(self, event: CallbackEvent) -> None:
        """Handle a SELECT_CAR event."""
        payload = event.payload
        car_id = payload.get("car_id", "unknown")
        location = payload.get("location", "")
        pickup_date = payload.get("pickup_date", "")
        dropoff_date = payload.get("dropoff_date", "")
        available_cars = payload.get("available_cars", [])

        print(f"\n[CarBookingAgent] Car {car_id} has been selected.")
        if available_cars:
            print("Available car options:")
            print(format_car_options(available_cars))
        print(f"Location: {location}, Pickup: {pickup_date}, Dropoff: {dropoff_date}")

        # Use the prompt template
        if available_cars:
            prompt = CAR_TYPE_SELECTION_PROMPT.format(
                options=format_car_options(available_cars),
                count=len(available_cars),
            )
            print(f"\n{prompt}")

        self.waiting_for_input = True

    def _handle_agent_completed_event(self, event: CallbackEvent) -> None:
        """Handle an AGENT_COMPLETED event."""
        agent_name = event.payload.get("agent", "Unknown")
        result = event.payload.get("result", {})
        final_answer = result.get("final_answer", "")

        if final_answer:
            # Extract just the final answer part
            if "Final Answer:" in final_answer:
                answer = final_answer.split("Final Answer:")[-1].strip()
                print(f"\n[{agent_name}] {answer}")
            else:
                print(f"\n[{agent_name}] {final_answer}")
        else:
            print(f"\n[{agent_name}] Task completed.")

    def _handle_booking_confirmed_event(self, event: CallbackEvent) -> None:
        """Handle a BOOKING_CONFIRMED event."""
        details = event.payload.get("details", {})
        print(f"\n[System] ✅ Booking confirmed!")
        print(json.dumps(details, indent=2))

    def _handle_booking_failed_event(self, event: CallbackEvent) -> None:
        """Handle a BOOKING_FAILED event."""
        reason = event.payload.get("reason", "Unknown error")
        print(f"\n[System] ❌ Booking failed: {reason}")

    def _handle_error_event(self, event: CallbackEvent) -> None:
        """Handle an ERROR event."""
        message = event.payload.get("message", "Unknown error")
        print(f"\n[System] ⚠️  Error: {message}")

    def _handle_events(self, events: list[CallbackEvent]) -> None:
        """Handle events emitted by the processor."""
        for event in events:
            if event.type == CallbackEventType.AWAITING_USER_INPUT:
                self._handle_awaiting_input_event(event)
            elif event.type == CallbackEventType.SELECT_CAR:
                self._handle_select_car_event(event)
            elif event.type == CallbackEventType.AGENT_COMPLETED:
                self._handle_agent_completed_event(event)
            elif event.type == CallbackEventType.BOOKING_CONFIRMED:
                self._handle_booking_confirmed_event(event)
            elif event.type == CallbackEventType.BOOKING_FAILED:
                self._handle_booking_failed_event(event)
            elif event.type == CallbackEventType.ERROR:
                self._handle_error_event(event)
            elif event.type == CallbackEventType.CAR_TYPE_CONFIRMED:
                car_id = event.payload.get("car_id", "unknown")
                car_type = event.payload.get("car_type", "")
                print(f"\n[System] Car type '{car_type}' confirmed for {car_id}.")

    def handle_user_input(self, user_input: str) -> None:
        """Process user input and handle the response."""
        if self.waiting_for_input:
            # This is a response to a human-in-the-loop prompt
            events = self.processor.process_user_response(user_input)
            self.waiting_for_input = False
        else:
            # This is a new user request
            events = self.processor.process_user_request(user_input)

        # Process any emitted events
        self._handle_events(events)

        # Also check the callback handler's event queue
        handler = self.processor.callback_handler
        while handler.has_events():
            event = handler.get_next_event()
            if event:
                self._handle_events([event])

    def run(self) -> None:
        """Run the CLI main loop."""
        self.print_header()

        while self.running:
            try:
                if self.waiting_for_input:
                    prompt = "> "
                else:
                    prompt = "\nYou: "

                user_input = input(prompt).strip()

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit"):
                    print("\nThank you for using the Travel Booking Agent System. Goodbye!")
                    self.running = False
                    break

                if user_input.lower() == "help":
                    self.print_help()
                    continue

                if user_input.lower() == "reset":
                    self.processor.reset()
                    self.waiting_for_input = False
                    print("\n[System] Conversation reset.")
                    continue

                self.handle_user_input(user_input)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                self.running = False
                break
            except EOFError:
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
    """Main entry point for the CLI."""
    cli = TravelBookingCLI()
    cli.run()


if __name__ == "__main__":
    main()