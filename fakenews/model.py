from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import networkx as nx
from mesa import Model
from mesa.space import NetworkGrid

from .agents import AgentParams, SocialAgent
from .constants import (
    ROLE_BOT,
    ROLE_FACT_CHECKER,
    ROLE_INFLUENCER,
    ROLE_REGULAR,
    STATE_BELIEVER,
    STATE_INFORMED,
    STATE_SUSCEPTIBLE,
)
from .news import FakeNewsClassifier, FakeNewsDataset, NewsItem


class FakeNewsModel(Model):
    """Model using a Barabasi-Albert graph and simple belief dynamics."""

    def __init__(
        self,
        n_agents: int = 200,
        m_links: int = 3,
        influencer_ratio: float = 0.05,
        bot_ratio: float = 0.03,
        fact_checker_ratio: float = 0.03,
        susceptible_ratio: float = 1.0,
        believer_ratio: float = 0.0,
        informed_ratio: float = 0.0,
        seed: int = 42,
        news_dataset: Optional[FakeNewsDataset] = None,
        train_classifier: bool = True,
        classifier_weight: float = 0.0,
        train_samples_per_agent: int = 0,
    ) -> None:
        super().__init__()
        random.seed(seed)
        self.num_agents = n_agents
        self.seed = seed

        self.graph = nx.barabasi_albert_graph(n_agents, m_links, seed=seed)
        self.grid = NetworkGrid(self.graph)
        self.agent_list: List[SocialAgent] = []

        # Global dynamics parameters (simple constants for PoC).
        self.belief_upper = 0.7
        self.belief_lower = 0.3
        self.believer_influence = 0.08
        self.informed_influence = 0.08
        self.bot_influence = 0.12
        self.fact_checker_influence = 0.12
        self.bot_post_prob = 1

        self.news_dataset: Optional[FakeNewsDataset] = news_dataset
        self.news_classifier: Optional[FakeNewsClassifier] = None
        self.train_dataset: Optional[FakeNewsDataset] = None
        self.test_dataset: Optional[FakeNewsDataset] = None
        self.classifier_weight = max(0.0, float(classifier_weight))
        self.train_samples_per_agent = max(0, int(train_samples_per_agent))
        self.initial_state_distribution = self._normalize_state_distribution(
            susceptible_ratio,
            believer_ratio,
            informed_ratio,
        )
        if news_dataset is not None and train_classifier:
            self.train_dataset, self.test_dataset = self._split_dataset(news_dataset)
            # If per-agent training is disabled, keep the old shared classifier behavior.
            if self.train_samples_per_agent <= 0:
                self.news_classifier = FakeNewsClassifier()
                self.news_classifier.train(self.train_dataset)
            else:
                self.news_classifier = None

        self._init_agents(influencer_ratio, bot_ratio, fact_checker_ratio)

    def _normalize_state_distribution(
        self,
        susceptible_ratio: float,
        believer_ratio: float,
        informed_ratio: float,
    ) -> Tuple[float, float, float]:
        raw_values = (
            max(0.0, float(susceptible_ratio)),
            max(0.0, float(believer_ratio)),
            max(0.0, float(informed_ratio)),
        )
        total = math.fsum(raw_values)
        if total <= 0.0:
            raise ValueError("State ratios must sum to a positive value")
        return tuple(value / total for value in raw_values)
    
    def _split_dataset(self, dataset: FakeNewsDataset, test_ratio: float = 0.1) -> Tuple[FakeNewsDataset, FakeNewsDataset]:
        items = dataset.items
        random.shuffle(items)
        split_idx = int(len(items) * (1 - test_ratio))
        train_items = items[:split_idx]
        test_items = items[split_idx:]
        return FakeNewsDataset(train_items), FakeNewsDataset(test_items)

    def _init_agents(self, influencer_ratio: float, bot_ratio: float, fact_checker_ratio: float) -> None:
        degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
        num_influencers = max(0, int(self.num_agents * influencer_ratio))
        influencer_nodes = {node for node, _deg in degrees[:num_influencers]}

        num_bots = max(0, int(self.num_agents * bot_ratio))
        num_fact_checkers = max(0, int(self.num_agents * fact_checker_ratio))
        remaining_nodes = [n for n in self.graph.nodes if n not in influencer_nodes]
        random.shuffle(remaining_nodes)
        bot_nodes = set(remaining_nodes[:num_bots])
        fact_checker_nodes = set(remaining_nodes[num_bots : num_bots + num_fact_checkers])
        susceptible_ratio, believer_ratio, informed_ratio = self.initial_state_distribution

        fake_items: Sequence[NewsItem] = []
        real_items: Sequence[NewsItem] = []
        if self.test_dataset is not None:
            fake_items = self.test_dataset.filter_label("FAKE")
            real_items = self.test_dataset.filter_label("REAL")

        # Per-agent classifier training chunks.
        train_fake: List[NewsItem] = []
        train_real: List[NewsItem] = []
        next_training_chunk = None
        if self.train_samples_per_agent > 0 and getattr(self, "train_dataset", None) is not None:
            train_fake = list(self.train_dataset.filter_label("FAKE"))
            train_real = list(self.train_dataset.filter_label("REAL"))
            rng = random.Random(self.seed)
            rng.shuffle(train_fake)
            rng.shuffle(train_real)
            fake_cursor = 0
            real_cursor = 0

            def _next_training_chunk() -> FakeNewsDataset:
                nonlocal fake_cursor, real_cursor
                # Try to include both classes to avoid degenerate single-class training.
                k_fake = self.train_samples_per_agent // 2
                k_real = self.train_samples_per_agent - k_fake

                chunk: List[NewsItem] = []
                for _ in range(k_fake):
                    if not train_fake:
                        break
                    if fake_cursor >= len(train_fake):
                        rng.shuffle(train_fake)
                        fake_cursor = 0
                    chunk.append(train_fake[fake_cursor])
                    fake_cursor += 1

                for _ in range(k_real):
                    if not train_real:
                        break
                    if real_cursor >= len(train_real):
                        rng.shuffle(train_real)
                        real_cursor = 0
                    chunk.append(train_real[real_cursor])
                    real_cursor += 1

                # If one of the classes is missing, top up from whatever exists.
                while len(chunk) < self.train_samples_per_agent:
                    if train_fake and (not train_real or rng.random() < 0.5):
                        if fake_cursor >= len(train_fake):
                            rng.shuffle(train_fake)
                            fake_cursor = 0
                        chunk.append(train_fake[fake_cursor])
                        fake_cursor += 1
                    elif train_real:
                        if real_cursor >= len(train_real):
                            rng.shuffle(train_real)
                            real_cursor = 0
                        chunk.append(train_real[real_cursor])
                        real_cursor += 1
                    else:
                        break

                rng.shuffle(chunk)
                return FakeNewsDataset(chunk)

            next_training_chunk = _next_training_chunk

        for node_id in self.graph.nodes:
            news_item: Optional[NewsItem] = None

            if node_id in bot_nodes:
                role = ROLE_BOT
                state = STATE_BELIEVER
                if fake_items:
                    news_item = random.choice(fake_items)
            elif node_id in fact_checker_nodes:
                role = ROLE_FACT_CHECKER
                state = STATE_INFORMED
                if real_items:
                    news_item = random.choice(real_items)
            elif node_id in influencer_nodes:
                role = ROLE_INFLUENCER
                state = self._pick_initial_state(susceptible_ratio, believer_ratio, informed_ratio)
            else:
                role = ROLE_REGULAR
                state = self._pick_initial_state(susceptible_ratio, believer_ratio, informed_ratio)

            params = self._sample_params(role)

            agent_classifier: Optional[FakeNewsClassifier] = None
            if self.train_samples_per_agent > 0 and next_training_chunk is not None:
                agent_classifier = FakeNewsClassifier()
                agent_classifier.train(next_training_chunk())

            agent = SocialAgent(
                node_id,
                self,
                role=role,
                state=state,
                params=params,
                news_item=news_item,
                news_classifier=agent_classifier,
            )
            self.agent_list.append(agent)
            self.grid.place_agent(agent, node_id)

    def _pick_initial_state(
        self,
        susceptible_ratio: float,
        believer_ratio: float,
        informed_ratio: float,
    ) -> str:
        draw = random.random()
        believer_cutoff = susceptible_ratio + believer_ratio
        informed_cutoff = believer_cutoff + informed_ratio

        if draw < susceptible_ratio:
            return STATE_SUSCEPTIBLE
        if draw < believer_cutoff:
            return STATE_BELIEVER
        if draw < informed_cutoff:
            return STATE_INFORMED

        # Guard against floating-point edge cases after normalization.
        return STATE_SUSCEPTIBLE

    def _sample_params(self, role: str) -> AgentParams:
        belief = random.uniform(0.4, 0.6)
        competence = random.uniform(0.2, 0.8)
        stubbornness = random.uniform(0.1, 0.7)
        fact_check_prob = random.uniform(0.05, 0.3)
        forgetting_prob = random.uniform(0.01, 0.05)
        engagement = random.uniform(0.2, 0.9)
        if role == ROLE_INFLUENCER:
            engagement = min(1.0, engagement + 0.2)
        if role == ROLE_BOT:
            engagement = 1.0

        trust = {
            ROLE_REGULAR: random.uniform(0.3, 0.7),
            ROLE_INFLUENCER: random.uniform(0.4, 0.9),
            ROLE_FACT_CHECKER: random.uniform(0.4, 0.9),
            ROLE_BOT: random.uniform(0.1, 0.6),
        }

        return AgentParams(
            belief=belief,
            competence=competence,
            stubbornness=stubbornness,
            fact_check_prob=fact_check_prob,
            forgetting_prob=forgetting_prob,
            engagement=engagement,
            trust=trust,
            homophily_threshold=random.uniform(0.1, 0.6),
        )

    def count_state(self, state: str) -> int:
        return sum(1 for a in self.agent_list if a.state == state)

    def step(self) -> None:
        agents = list(self.agent_list)
        random.shuffle(agents)
        for agent in agents:
            agent.step()
