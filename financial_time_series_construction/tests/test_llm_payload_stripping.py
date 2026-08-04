"""Verify time series data is not passed to the LLM."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from financial_time_series_construction.handler import TimeSeriesConstructionHandler
from financial_time_series_construction.models import LLMRequest, ModelRequestFactory
from financial_time_series_construction.processor import TimeSeriesConstructionProcessor
from financial_time_series_construction.tools import set_run_id

logger = logging.getLogger(__name__)


class CapturingModelFactory(ModelRequestFactory):
    """Captures all LLM messages and returns canned responses."""

    def __init__(self) -> None:
        self.captured_messages: list[list[dict[str, str]]] = []
        self.call_count = 0

    def chat(self, request: LLMRequest) -> str:
        self.captured_messages.append(list(request.messages))
        self.call_count += 1
        system_prompt = request.system_prompt or ""
        all_text = " ".join(m.get("content", "") for m in request.messages).casefold()

        if "reference" in system_prompt.casefold() or "instrument" in all_text:
            return 'Thought: resolve.\nAction: get_instrument_details\nAction Input: {"query": "AAPL"}\n'
        if "market" in system_prompt.casefold() or "historical" in all_text:
            return "Thought: load.\nAction: available_data_sources\nAction Input: {}\n"
        if "quality" in system_prompt.casefold() or "check_data_quality" in all_text:
            return 'Thought: check.\nAction: check_data_quality\nAction Input: {"source": "yahoo", "symbol": "AAPL"}\n'
        return "Thought: done.\nFinal Answer: Workflow completed."


@pytest.fixture
def mock_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create mock CSV data files and patch tools.DATA_DIR."""
    import financial_time_series_construction.tools as tools_module

    set_run_id("test_llm_payload")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    instruments = pd.DataFrame({
        "symbol": ["AAPL"],
        "security_name": ["Apple Inc."],
        "sector": ["IT"],
        "sub_industry": ["Hardware"],
        "date_added": ["1982-11-30"],
    })
    instruments.to_csv(data_dir / "instruments.csv", index=False)

    dates = pd.bdate_range("2023-01-01", "2024-12-31")
    n = len(dates)
    for source_name, base in [("yahoo", 150.0), ("bloomberg", 151.0), ("reuters", 149.0)]:
        prices = [base + i * 0.05 for i in range(n)]
        df = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "AAPL": prices})
        df.to_csv(data_dir / f"{source_name}_stock_data.csv", index=False)

    monkeypatch.setattr(tools_module, "DATA_DIR", data_dir)
    return data_dir


def test_time_series_data_not_passed_to_llm(mock_data_dir: Path) -> None:
    """Run workflow and verify no large dates/prices arrays reach the LLM."""
    factory = CapturingModelFactory()
    handler = TimeSeriesConstructionHandler(session_id="test_llm_payload")
    processor = TimeSeriesConstructionProcessor(factory=factory, handler=handler)

    events = processor.process_user_request("Build AAPL from 2023-01-01 to 2024-01-01")
    assert events, "Expected events from workflow"

    large_array_found = False
    for i, messages in enumerate(factory.captured_messages):
        text = "\n".join(m.get("content", "") for m in messages)
        for line in text.split("\n"):
            if '"dates": [' in line or '"prices": [' in line:
                start = line.find("[")
                end = line.rfind("]")
                if start >= 0 and end > start:
                    elements = [e for e in line[start + 1 : end].split(",") if e.strip()]
                    if len(elements) > 10:
                        large_array_found = True
                        logger.error("LLM msg %d has large array: %s...", i, line[:200])

    assert not large_array_found, "Large time series arrays were passed to the LLM!"
    logger.info("Verified: %d LLM calls, no large time series arrays passed to LLM.", factory.call_count)