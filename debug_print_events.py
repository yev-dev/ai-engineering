from financial_time_series_construction.agents_definition import CallbackEventType
from financial_time_series_construction.processor import TimeSeriesConstructionProcessor


class SequenceFactory:
    def __init__(self, seq: list[str]) -> None:
        self.chat_sequence = list(seq)

    def chat(self, request):
        return self.chat_sequence.pop(0)


def dump_events(label: str, events: list) -> None:
    print(f"\n=== {label} ({len(events)} events) ===")
    for idx, event in enumerate(events):
        payload = event.payload
        print(f"{idx:02d} {event.type.value} agent={payload.get('agent')}")
        if event.type == CallbackEventType.ERROR:
            print(f"   error_payload={payload}")
    awaiting = [e for e in events if e.type == CallbackEventType.AWAITING_USER_INPUT]
    print(f"AWAITING count: {len(awaiting)}")
    if awaiting:
        print(f"AWAITING payload: {awaiting[-1].payload}")


def main() -> None:
    chat_sequence = [
        "Action: get_instrument_details\nAction Input: {\"query\": \"APL\"}",
        "Action: delegate_to_agent\nAction Input: {\"agent_name\": \"MarketDataAgent\", \"request\": \"load AAPL from 2023-01-01 to 2024-12-31 from all sources\"}",
        "Action: available_data_sources\nAction Input: {}",
        "Action: historical_prices\nAction Input: {\"symbol\": \"AAPL\", \"start_date\": \"2023-01-03\", \"end_date\": \"2024-12-30\", \"source\": \"yahoo\"}",
        "Action: delegate_to_agent\nAction Input: {\"agent_name\": \"DataQualityAgent\", \"request\": \"check quality of AAPL from yahoo\"}",
        "Action: check_data_quality\nAction Input: {\"prices\": [150.0, 150.5, 151.0], \"source\": \"yahoo\", \"symbol\": \"AAPL\"}",
        "Action: delegate_to_agent\nAction Input: {\"agent_name\": \"ReportingAgent\", \"request\": \"present quality report for AAPL from yahoo\"}",
        "Final Answer: User selected yahoo as the data source for AAPL.",
        "Final Answer: Gap filling can now proceed once a method is selected.",
    ]

    processor = TimeSeriesConstructionProcessor(factory=SequenceFactory(chat_sequence))

    events = processor.process_user_request("create a time series for APL between 2023 and 2024")
    dump_events("request", events)
    print(
        "handler state after request:",
        {
            "waiting_for_input": processor.handler.waiting_for_input,
            "current_agent": processor.handler.current_agent,
            "paused_agent": (processor.handler.paused_state or {}).get("agent"),
            "checkpoint": (processor.handler.paused_state or {}).get("checkpoint"),
        },
    )

    if processor.handler.waiting_for_input:
        resume_events = processor.process_user_response("yahoo")
        dump_events("resume(yahoo)", resume_events)


if __name__ == "__main__":
    main()
