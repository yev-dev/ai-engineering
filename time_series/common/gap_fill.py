from __future__ import annotations

import pandas as pd

from .models import GapFillSuggestion


class GapFillAgent:
    """Suggests and applies gap-filling methods for missing historical prices."""

    def suggest(self, series: pd.Series) -> list[GapFillSuggestion]:
        missing_ratio = float(series.isna().mean()) if len(series) else 0.0
        if missing_ratio <= 0.02:
            primary = GapFillSuggestion(
                method="linear",
                rationale="Low missingness; linear interpolation preserves local trend and smoothness.",
            )
        elif missing_ratio <= 0.1:
            primary = GapFillSuggestion(
                method="ffill_then_bfill",
                rationale="Moderate missingness; directional carry-forward with fallback is robust for prices.",
            )
        else:
            primary = GapFillSuggestion(
                method="rolling_median",
                rationale="Higher missingness; rolling median is more robust to noise and outliers.",
            )

        return [
            primary,
            GapFillSuggestion(method="linear", rationale="Interpolates between neighboring observations."),
            GapFillSuggestion(method="ffill_then_bfill", rationale="Carries last known value forward and fills edges."),
            GapFillSuggestion(method="rolling_median", rationale="Fills with median of a rolling window for robustness."),
        ]

    def apply(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        out = df.sort_values("date").copy()
        if method == "linear":
            out["close"] = out["close"].interpolate(method="linear").bfill().ffill()
        elif method == "ffill_then_bfill":
            out["close"] = out["close"].ffill().bfill()
        elif method == "rolling_median":
            filled = out["close"].copy()
            median = out["close"].rolling(window=5, min_periods=1).median()
            filled = filled.fillna(median)
            out["close"] = filled.ffill().bfill()
        else:
            raise ValueError(f"Unsupported gap-fill method: {method}")
        return out
