from __future__ import annotations

import random
from typing import List, Optional, Sequence

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
        seed: int = 42,
        news_dataset: Optional[FakeNewsDataset] = None,
        train_classifier: bool = True,
        classifier_weight: float = 0.0,
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
        self.bot_post_prob = 0.6

        self.news_dataset: Optional[FakeNewsDataset] = news_dataset
        self.news_classifier: Optional[FakeNewsClassifier] = None
        self.classifier_weight = max(0.0, float(classifier_weight))
        if news_dataset is not None and train_classifier:
            self.train_dataset, self.test_dataset = self._split_dataset(news_dataset)
            self.news_classifier = FakeNewsClassifier()
            self.news_classifier.train(self.train_dataset)

        self._init_agents(influencer_ratio, bot_ratio, fact_checker_ratio)
    
    def _split_dataset(self, dataset: FakeNewsDataset, test_ratio: float = 0.1) -> Tuple[FakeNewsDataset, FakeNewsDataset]:
        items = dataset.items
        random.shuffle(items)
        split_idx = int(len(items) * (1 - test_ratio))
        train_items = items[:split_idx]
        test_items = items[split_idx:]
        return FakeNewsDataset(train_items), FakeNewsDataset(test_items)

    def _init_agents(self, influencer_ratio: float, bot_ratio: float, fact_checker_ratio: float) -> None:
        degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
        num_influencers = max(1, int(self.num_agents * influencer_ratio))
        influencer_nodes = {node for node, _deg in degrees[:num_influencers]}

        num_bots = max(1, int(self.num_agents * bot_ratio))
        num_fact_checkers = max(1, int(self.num_agents * fact_checker_ratio))
        remaining_nodes = [n for n in self.graph.nodes if n not in influencer_nodes]
        random.shuffle(remaining_nodes)
        bot_nodes = set(remaining_nodes[:num_bots])
        fact_checker_nodes = set(remaining_nodes[num_bots : num_bots + num_fact_checkers])

        fake_items: Sequence[NewsItem] = []
        real_items: Sequence[NewsItem] = []
        if self.test_dataset is not None:
            fake_items = self.test_dataset.filter_label("FAKE")
            real_items = self.test_dataset.filter_label("REAL")

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
                state = STATE_SUSCEPTIBLE
            else:
                role = ROLE_REGULAR
                state = STATE_SUSCEPTIBLE

            params = self._sample_params(role)
            agent = SocialAgent(node_id, self, role=role, state=state, params=params, news_item=news_item)
            self.agent_list.append(agent)
            self.grid.place_agent(agent, node_id)

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
