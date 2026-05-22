from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import SocialAgent
from .constants import (
    ROLE_BOT,
    ROLE_FACT_CHECKER,
    ROLE_INFLUENCER,
    ROLE_REGULAR,
    STATE_BELIEVER,
    STATE_INFORMED,
    STATE_SUSCEPTIBLE,
)
from .model import FakeNewsModel


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class SimulationResultsRecorder:
    """Collects and writes simulation outputs:

    - [timestamp]_parameters.json
    - [timestamp]_timeseries.csv
    - [timestamp]_graph.csv
    """

    def __init__(self, output_dir: str | Path, run_timestamp: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = run_timestamp or make_timestamp()

        self.parameters: Dict[str, Any] = {
            "run_timestamp": self.timestamp,
        }
        self.timeseries_rows: List[Dict[str, Any]] = []
        self.graph_rows: List[Dict[str, Any]] = []

    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.parameters.update(params)

    def record_step(self, time_step: int, model: FakeNewsModel) -> None:
        self.timeseries_rows.append(
            {
                "time_step": time_step,
                "susceptible": model.count_state(STATE_SUSCEPTIBLE),
                "believer": model.count_state(STATE_BELIEVER),
                "informed": model.count_state(STATE_INFORMED),
            }
        )

        for agent in model.agent_list:
            self.graph_rows.append(self._agent_row(model, time_step, agent))

    def _agent_row(self, model: FakeNewsModel, time_step: int, agent: SocialAgent) -> Dict[str, Any]:
        # Flatten trust dict into columns for CSV.
        trust = agent.params.trust
        news_item = agent.news_item
        pred = agent.news_prediction

        return {
            "run_timestamp": self.timestamp,
            "time_step": time_step,
            "agent_id": agent.unique_id,
            "agent_role": agent.role,
            "agent_state": agent.state,
            "belief": agent.params.belief,
            "competence": agent.params.competence,
            "stubbornness": agent.params.stubbornness,
            "fact_check_prob": agent.params.fact_check_prob,
            "forgetting_prob": agent.params.forgetting_prob,
            "engagement": agent.params.engagement,
            "homophily_threshold": agent.params.homophily_threshold,
            "trust_regular": trust.get(ROLE_REGULAR),
            "trust_influencer": trust.get(ROLE_INFLUENCER),
            "trust_fact_checker": trust.get(ROLE_FACT_CHECKER),
            "trust_bot": trust.get(ROLE_BOT),
            "news_id": (news_item.news_id if news_item else None),
            "news_title": (news_item.title if news_item else None),
            "news_text": (news_item.text if news_item else None),
            "news_true_label": (news_item.label if news_item else None),
            "news_pred_label": (pred.predicted_label if pred else None),
            "news_pred_proba_fake": (pred.proba_fake if pred else None),
            "classifier_trained": bool(getattr(model.news_classifier, "is_trained", False)) if model.news_classifier else False,
        }

    def write_files(self) -> Dict[str, Path]:
        paths = {
            "parameters": self.output_dir / f"{self.timestamp}_parameters.json",
            "timeseries": self.output_dir / f"{self.timestamp}_timeseries.csv",
            "graph": self.output_dir / f"{self.timestamp}_graph.csv",
        }

        paths["parameters"].write_text(json.dumps(self.parameters, indent=2, ensure_ascii=False), encoding="utf-8")

        self._write_csv(paths["timeseries"], self.timeseries_rows)
        self._write_csv(paths["graph"], self.graph_rows)

        return paths

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=";",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            writer.writerows(rows)
