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
from .news import FakeNewsClassifier, NewsItem, NewsPrediction, predict_news


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
        news_classifier: Optional[FakeNewsClassifier] = None,
    ) -> None:
        super().__init__(model)
        self.unique_id = unique_id
        self.role = role
        self.state = state
        self.params = params

        self.news_item: Optional[NewsItem] = news_item
        self.news_prediction: Optional[NewsPrediction] = None
        self.news_classifier: Optional[FakeNewsClassifier] = news_classifier

        # When an agent decides to forward a news item, do it in their own step.
        # This avoids recursive broadcast cascades inside a single influence call.
        self._pending_share_item: Optional[NewsItem] = None

    def step(self) -> None:
        if self.role == ROLE_BOT:
            self._bot_step()
            return
        if self.role == ROLE_FACT_CHECKER:
            self._fact_checker_step()
            return

        self._interact_with_neighbors()
        self._share_pending_news()
        self._update_state()

    def _share_pending_news(self) -> None:
        if self._pending_share_item is None:
            return

        item = self._pending_share_item
        self._pending_share_item = None

        # Minimal forwarding rule: if this agent currently believes the item is real
        # (including FAKE items misclassified as REAL), share it with neighbors.
        self._broadcast_influence(+self.model.believer_influence, source_role=self.role, news_item=item)

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
            self._apply_classifier_influence(news_item)

    def _apply_classifier_influence(self, news_item: NewsItem) -> None:
        classifier = self.news_classifier or getattr(self.model, "news_classifier", None)
        news_prediction = predict_news(classifier, news_item)

        # Store for results/plots even if weight==0.
        self.news_prediction = news_prediction
        if news_prediction is None:
            return

        # Forwarding rule for non-bot / non-fact-checker agents:
        # - If the agent predicts the item as REAL, they forward it.
        # - If the item is FAKE and correctly predicted as FAKE, they do not forward it.
        if self.role not in {ROLE_BOT, ROLE_FACT_CHECKER}:
            if news_prediction.predicted_label == "REAL":
                self._pending_share_item = news_item
            elif news_item.label.upper() == "FAKE" and news_prediction.predicted_label == "FAKE":
                # Correctly recognized fake -> don't forward it.
                if self._pending_share_item == news_item:
                    self._pending_share_item = None

        weight = float(getattr(self.model, "classifier_weight", 0.0))
        if weight <= 0.0:
            return
        
        if news_prediction.predicted_label == "REAL":
            self.news_item = news_item

        direction = -1.0 if news_prediction.predicted_label == news_item.label.upper() else 1.0
        confidence = news_prediction.proba_fake if news_prediction.predicted_label == "FAKE" else 1.0 - news_prediction.proba_fake
        if direction < 0 and self.params.belief > random.random() * confidence * self.params.competence * 4:
            direction *= -1.0
        self.params.belief = max(0.0, min(1.0, self.params.belief + direction * weight * confidence))

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
