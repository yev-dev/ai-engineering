#!/usr/bin/env python3
"""
CLI for the travel booking agent system.

Provides an interactive command-line interface where users can:
  - Make travel requests in natural language
  - Respond to agent prompts (human-in-the-loop)
  - Confirm car types
  - View booking status

Usage:
    python cli.py
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from agents import Event, EventType
from handler import process_event
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
    """Interactive CLI for the travel booking system."""

    def __init__(self) -> None:
        self.processor = TravelBookingProcessor()
        self.running = True
        self.waiting_for_input = False
        self.pending_context: dict[str, Any] | None = None

    def print_header(self) -> None:
        """Print the application header."""
        print("=" * 60)
        print("  Travel Booking Agent System")
        print("  Powered by ReAct Agents + Ollama (gemma4:e4b)")
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

    def handle_user_input(self, user_input: str) -> None:
        """Process user input and handle the response."""
        if self.waiting_for_input:
            # This is a response to a human-in-the-loop prompt
            events = self.processor.process_user_response(
                user_input=user_input,
                context=self.pending_context,
            )
            self.waiting_for_input = False
            self.pending_context = None
        else:
            # This is a new user request
            events = self.processor.process_user_request(user_input)

        # Process any emitted events
        self._handle_events(events)

    def _handle_events(self, events: list[Event]) -> None:
        """Handle events emitted by the processor."""
        for event in events:
            if event.type == EventType.AWAITING_USER_INPUT:
                self.waiting_for_input = True
                self.pending_context = event.payload
                # The handler already printed the prompt
            elif event.type == EventType.SELECT_CAR:
                self.waiting_for_input = True
                self.pending_context = {
                    "event_type": EventType.SELECT_CAR.value,
                    "car_id": event.payload.get("car_id", ""),
                    "location": event.payload.get("location", ""),
                    "pickup_date": event.payload.get("pickup_date", ""),
                    "dropoff_date": event.payload.get("dropoff_date", ""),
                    "available_cars": event.payload.get("available_cars", []),
                    "agent": "CarBookingAgent",
                }
                print("\n[CarBookingAgent] Please confirm the car type you'd like:")
                print("  (Type the car model or 'cancel' to abort)")
            elif event.type == EventType.AGENT_COMPLETED:
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
            elif event.type == EventType.BOOKING_CONFIRMED:
                details = event.payload.get("details", {})
                print(f"\n[System] ✅ Booking confirmed!")
                print(json.dumps(details, indent=2))
            elif event.type == EventType.BOOKING_FAILED:
                reason = event.payload.get("reason", "Unknown error")
                print(f"\n[System] ❌ Booking failed: {reason}")
            elif event.type == EventType.ERROR:
                message = event.payload.get("message", "Unknown error")
                print(f"\n[System] ⚠️  Error: {message}")

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
                    self.pending_context = None
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