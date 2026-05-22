from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional

from mesa import Agent, Model

from .constants import (
    ROLE_BOT,
    ROLE_FACT_CHECKER,
    ROLE_REGULAR,
    STATE_BELIEVER,
    STATE_INFORMED,
    STATE_SUSCEPTIBLE,
)
from .news import NewsItem, NewsPrediction, predict_news


@dataclass
class AgentParams:
    belief: float
    competence: float
    stubbornness: float
    fact_check_prob: float
    forgetting_prob: float
    engagement: float
    trust: Dict[str, float]
    homophily_threshold: float


class SocialAgent(Agent):
    """A minimal agent with belief dynamics + optional held news item."""

    def __init__(
        self,
        unique_id: int,
        model: Model,
        role: str,
        state: str,
        params: AgentParams,
        news_item: Optional[NewsItem] = None,
    ) -> None:
        super().__init__(model)
        self.unique_id = unique_id
        self.role = role
        self.state = state
        self.params = params

        self.news_item: Optional[NewsItem] = news_item
        self.news_prediction: Optional[NewsPrediction] = None
        self._refresh_news_prediction()

    def _refresh_news_prediction(self) -> None:
        classifier = getattr(self.model, "news_classifier", None)
        self.news_prediction = predict_news(classifier, self.news_item)

    def step(self) -> None:
        if self.role == ROLE_BOT:
            self._bot_step()
            return
        if self.role == ROLE_FACT_CHECKER:
            self._fact_checker_step()
            return

        self._interact_with_neighbors()
        self._update_state()

    def _bot_step(self) -> None:
        # Bots only push misinformation; they never change state.
        if random.random() > self.model.bot_post_prob:
            return
        self._broadcast_influence(+self.model.bot_influence, source_role=ROLE_BOT, news_item=self.news_item)

    def _fact_checker_step(self) -> None:
        # Fact-checkers only push corrective info; they never change state.
        self._broadcast_influence(
            -self.model.fact_checker_influence,
            source_role=ROLE_FACT_CHECKER,
            news_item=self.news_item,
        )

    def _interact_with_neighbors(self) -> None:
        neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
        if not neighbors:
            return

        # Engagement controls how many interactions happen this step.
        k = max(1, int(round(self.params.engagement * len(neighbors))))
        sampled = random.sample(neighbors, k=min(k, len(neighbors)))

        for neighbor in sampled:
            if abs(self.params.belief - neighbor.params.belief) > self.params.homophily_threshold:
                continue

            if neighbor.state == STATE_BELIEVER or neighbor.role == ROLE_BOT:
                self._apply_influence(+self.model.believer_influence, neighbor.role, neighbor.news_item)
            elif neighbor.state == STATE_INFORMED or neighbor.role == ROLE_FACT_CHECKER:
                self._apply_influence(-self.model.informed_influence, neighbor.role, neighbor.news_item)

    def _broadcast_influence(self, base_influence: float, source_role: str, news_item: Optional[NewsItem]) -> None:
        neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
        for neighbor in neighbors:
            if abs(self.params.belief - neighbor.params.belief) > self.params.homophily_threshold:
                continue
            neighbor._apply_influence(base_influence, source_role, news_item)

    def _apply_influence(self, base_influence: float, source_role: str, news_item: Optional[NewsItem]) -> None:
        trust = self.params.trust.get(source_role, 0.5)
        resistance = 1.0 - self.params.stubbornness
        # Competence reduces susceptibility to misinformation only.
        if base_influence > 0:
            resistance *= 1.0 - self.params.competence

        delta = base_influence * trust * resistance
        self.params.belief = max(0.0, min(1.0, self.params.belief + delta))

        # Possible self-verification when exposed to correction.
        if base_influence < 0 and random.random() < self.params.fact_check_prob:
            self.params.belief = max(0.0, self.params.belief - abs(base_influence))

        # Adopt the content (textual fake/real news) being propagated.
        if news_item is not None:
            self.news_item = news_item
            self._refresh_news_prediction()
            self._apply_classifier_influence()

    def _apply_classifier_influence(self) -> None:
        weight = float(getattr(self.model, "classifier_weight", 0.0))
        if weight <= 0.0 or self.news_prediction is None:
            return

        # Push belief based on classifier confidence: FAKE => higher belief, REAL => lower belief.
        confidence = min(1.0, max(0.0, abs(self.news_prediction.proba_fake - 0.5) * 2.0))
        direction = 1.0 if self.news_prediction.predicted_label == "FAKE" else -1.0
        delta = weight * confidence * direction
        self.params.belief = max(0.0, min(1.0, self.params.belief + delta))

    def _update_state(self) -> None:
        if self.state == STATE_BELIEVER and random.random() < self.params.forgetting_prob:
            self.state = STATE_SUSCEPTIBLE
        elif self.state == STATE_INFORMED and random.random() < self.params.forgetting_prob:
            self.state = STATE_SUSCEPTIBLE

        if self.state == STATE_SUSCEPTIBLE:
            if self.params.belief >= self.model.belief_upper:
                self.state = STATE_BELIEVER
            elif self.params.belief <= self.model.belief_lower:
                self.state = STATE_INFORMED
        elif self.state == STATE_BELIEVER:
            if self.params.belief <= self.model.belief_lower:
                self.state = STATE_INFORMED
        elif self.state == STATE_INFORMED:
            if self.params.belief >= self.model.belief_upper:
                self.state = STATE_BELIEVER
