from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .models import QualityMetric, ReActEvent


def generate_run_id(prefix: str = "tsrun") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid4().hex[:8]}"


class AuditArtifactManager:
    def __init__(self, export_dir: str, run_id: str) -> None:
        self.export_root = Path(export_dir)
        self.run_id = run_id
        self.run_dir = self.export_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def export_dataframe(self, df: pd.DataFrame, filename: str) -> str:
        path = self.run_dir / filename
        df.to_csv(path, index=False)
        return str(path)

    def export_json(self, payload: dict, filename: str) -> str:
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return str(path)

    def build_run_report(
        self,
        framework: str,
        request: dict,
        selected_source: str,
        gap_method: str,
        quality: list[QualityMetric],
        react_events: list[ReActEvent],
        artifact_paths: dict[str, str],
    ) -> dict:
        return {
            "run_id": self.run_id,
            "framework": framework,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request": request,
            "selected_source": selected_source,
            "gap_method": gap_method,
            "quality": [asdict(q) for q in quality],
            "react_trace": [asdict(e) for e in react_events],
            "artifacts": artifact_paths,
        }
