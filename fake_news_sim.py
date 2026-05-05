"""Simple PoC simulation of fake news diffusion on a scale-free social graph."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid


STATE_SUSCEPTIBLE = "susceptible"
STATE_BELIEVER = "believer"
STATE_INFORMED = "informed"

ROLE_REGULAR = "regular"
ROLE_INFLUENCER = "influencer"
ROLE_FACT_CHECKER = "fact_checker"
ROLE_BOT = "bot"


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
    """A minimal agent with belief dynamics and state transitions."""

    def __init__(
        self,
        unique_id: int,
        model: Model,
        role: str,
        state: str,
        params: AgentParams,
    ) -> None:
        super().__init__(model)
        self.unique_id = unique_id
        self.role = role
        self.state = state
        self.params = params

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
        self._broadcast_influence(+self.model.bot_influence, source_role=ROLE_BOT)

    def _fact_checker_step(self) -> None:
        # Fact-checkers only push corrective info; they never change state.
        self._broadcast_influence(-self.model.fact_checker_influence, source_role=ROLE_FACT_CHECKER)

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
                self._apply_influence(+self.model.believer_influence, neighbor.role)
            elif neighbor.state == STATE_INFORMED or neighbor.role == ROLE_FACT_CHECKER:
                self._apply_influence(-self.model.informed_influence, neighbor.role)

    def _broadcast_influence(self, base_influence: float, source_role: str) -> None:
        neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
        for neighbor in neighbors:
            if abs(self.params.belief - neighbor.params.belief) > self.params.homophily_threshold:
                continue
            neighbor._apply_influence(base_influence, source_role)

    def _apply_influence(self, base_influence: float, source_role: str) -> None:
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


class FakeNewsModel(Model):
    """PoC model using a Barabasi-Albert graph and simple belief dynamics."""

    def __init__(
        self,
        n_agents: int = 200,
        m_links: int = 3,
        influencer_ratio: float = 0.05,
        bot_ratio: float = 0.03,
        fact_checker_ratio: float = 0.03,
        seed: int = 42,
    ) -> None:
        super().__init__()
        random.seed(seed)
        self.num_agents = n_agents
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

        self._init_agents(influencer_ratio, bot_ratio, fact_checker_ratio)

        self.datacollector = DataCollector(
            model_reporters={
                "susceptible": lambda m: self._count_state(STATE_SUSCEPTIBLE),
                "believer": lambda m: self._count_state(STATE_BELIEVER),
                "informed": lambda m: self._count_state(STATE_INFORMED),
            }
        )

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

        for node_id in self.graph.nodes:
            if node_id in bot_nodes:
                role = ROLE_BOT
                state = STATE_BELIEVER
            elif node_id in fact_checker_nodes:
                role = ROLE_FACT_CHECKER
                state = STATE_INFORMED
            elif node_id in influencer_nodes:
                role = ROLE_INFLUENCER
                state = STATE_SUSCEPTIBLE
            else:
                role = ROLE_REGULAR
                state = STATE_SUSCEPTIBLE

            params = self._sample_params(role)
            agent = SocialAgent(node_id, self, role=role, state=state, params=params)
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

    def _count_state(self, state: str) -> int:
        return sum(1 for a in self.agent_list if a.state == state)

    def step(self) -> None:
        self.datacollector.collect(self)
        agents = list(self.agent_list)
        random.shuffle(agents)
        for agent in agents:
            agent.step()


def run_simulation(
    steps: int = 50,
    n_agents: int = 200,
    m_links: int = 3,
    influencer_ratio: float = 0.05,
    bot_ratio: float = 0.03,
    fact_checker_ratio: float = 0.03,
    seed: int = 42,
) -> Tuple[FakeNewsModel, List[int], List[int], List[int]]:
    model = FakeNewsModel(
        n_agents=n_agents,
        m_links=m_links,
        influencer_ratio=influencer_ratio,
        bot_ratio=bot_ratio,
        fact_checker_ratio=fact_checker_ratio,
        seed=seed,
    )
    for _ in range(steps):
        model.step()

    data = model.datacollector.get_model_vars_dataframe()
    susceptible = data["susceptible"].tolist()
    believer = data["believer"].tolist()
    informed = data["informed"].tolist()
    return model, susceptible, believer, informed


def plot_timeseries(susceptible: List[int], believer: List[int], informed: List[int]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(susceptible, label="Susceptible")
    plt.plot(believer, label="Believer")
    plt.plot(informed, label="Informed")
    plt.title("Fake News Spreading")
    plt.xlabel("Step")
    plt.ylabel("Agents")
    plt.legend()
    plt.tight_layout()


def plot_graph(model: FakeNewsModel) -> None:
    import matplotlib.pyplot as plt

    color_map = {
        STATE_SUSCEPTIBLE: "#9aa0a6",
        STATE_BELIEVER: "#d93025",
        STATE_INFORMED: "#188038",
    }

    node_states = {agent.pos: agent.state for agent in model.agent_list}
    node_colors = [color_map.get(node_states.get(n, STATE_SUSCEPTIBLE)) for n in model.graph.nodes]

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(model.graph, seed=42)
    nx.draw_networkx_edges(model.graph, pos, alpha=0.15, width=0.6)
    nx.draw_networkx_nodes(model.graph, pos, node_color=node_colors, node_size=40)
    plt.title("Social Graph (colored by state)")
    plt.axis("off")


def animate_simulation(
    model: FakeNewsModel,
    steps: int,
    show_graph: bool = True,
    show_timeseries: bool = False,
    interval_ms: int = 200,
) -> object:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    color_map = {
        STATE_SUSCEPTIBLE: "#9aa0a6",
        STATE_BELIEVER: "#d93025",
        STATE_INFORMED: "#188038",
    }

    if show_graph and show_timeseries:
        fig, (ax_ts, ax_graph) = plt.subplots(1, 2, figsize=(12, 5))
    elif show_graph:
        fig, ax_graph = plt.subplots(1, 1, figsize=(8, 6))
        ax_ts = None
    else:
        fig, ax_ts = plt.subplots(1, 1, figsize=(8, 4))
        ax_graph = None

    ts_susceptible: List[int] = []
    ts_believer: List[int] = []
    ts_informed: List[int] = []

    line_sus = line_bel = line_inf = None
    if ax_ts is not None:
        line_sus, = ax_ts.plot([], [], label="Susceptible")
        line_bel, = ax_ts.plot([], [], label="Believer")
        line_inf, = ax_ts.plot([], [], label="Informed")
        ax_ts.set_title("Fake News Spreading")
        ax_ts.set_xlabel("Step")
        ax_ts.set_ylabel("Agents")
        ax_ts.legend()

    pos = nx.spring_layout(model.graph, seed=42)
    graph_nodes = None
    graph_edges = None

    if ax_graph is not None:
        ax_graph.set_title("Social Graph (colored by state)")
        ax_graph.axis("off")
        graph_edges = nx.draw_networkx_edges(model.graph, pos, ax=ax_graph, alpha=0.15, width=0.6)
        node_colors = [color_map.get(STATE_SUSCEPTIBLE) for _ in model.graph.nodes]
        graph_nodes = nx.draw_networkx_nodes(
            model.graph,
            pos,
            ax=ax_graph,
            node_color=node_colors,
            node_size=40,
        )

    def update(frame_index: int) -> Tuple[object, ...]:
        model.step()
        ts_susceptible.append(model._count_state(STATE_SUSCEPTIBLE))
        ts_believer.append(model._count_state(STATE_BELIEVER))
        ts_informed.append(model._count_state(STATE_INFORMED))

        artists: List[object] = []
        if ax_ts is not None and line_sus is not None and line_bel is not None and line_inf is not None:
            x = list(range(len(ts_susceptible)))
            line_sus.set_data(x, ts_susceptible)
            line_bel.set_data(x, ts_believer)
            line_inf.set_data(x, ts_informed)
            ax_ts.set_xlim(0, max(10, len(ts_susceptible)))
            ax_ts.set_ylim(0, model.num_agents)
            artists.extend([line_sus, line_bel, line_inf])

        if graph_nodes is not None:
            node_states = {agent.pos: agent.state for agent in model.agent_list}
            node_colors = [color_map.get(node_states.get(n, STATE_SUSCEPTIBLE)) for n in model.graph.nodes]
            graph_nodes.set_color(node_colors)
            if graph_edges is not None:
                artists.append(graph_edges)
            artists.append(graph_nodes)
            if ax_graph is not None:
                ax_graph.set_title(
                    f"Step {frame_index + 1} | S:{ts_susceptible[-1]} B:{ts_believer[-1]} I:{ts_informed[-1]}"
                )

        return tuple(artists)

    animation = FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False, repeat=False)
    # Render the first step immediately so the window does not look frozen.
    update(0)
    plt.tight_layout()
    plt.show()
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake news spreading")
    parser.add_argument("--agents", type=int, default=200, help="Number of agents")
    parser.add_argument("--steps", type=int, default=60, help="Number of simulation steps")
    parser.add_argument("--m-links", type=int, default=3, help="BA model links per new node")
    parser.add_argument("--influencers", type=float, default=0.05, help="Influencer ratio")
    parser.add_argument("--bots", type=float, default=0.03, help="Bot ratio")
    parser.add_argument("--fact-checkers", type=float, default=0.03, help="Fact-checker ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph visualization")
    parser.add_argument("--no-animate", action="store_true", help="Disable animation and show static plots")
    parser.add_argument("--interval", type=int, default=200, help="Animation frame interval in ms")
    parser.add_argument("--timeseries", action="store_true", help="Show time series alongside graph")
    args = parser.parse_args()

    if args.no_animate:
        model, susceptible, believer, informed = run_simulation(
            steps=args.steps,
            n_agents=args.agents,
            m_links=args.m_links,
            influencer_ratio=args.influencers,
            bot_ratio=args.bots,
            fact_checker_ratio=args.fact_checkers,
            seed=args.seed,
        )

        plot_timeseries(susceptible, believer, informed)
        if not args.no_graph:
            plot_graph(model)

        import matplotlib.pyplot as plt

        plt.show()
    else:
        model = FakeNewsModel(
            n_agents=args.agents,
            m_links=args.m_links,
            influencer_ratio=args.influencers,
            bot_ratio=args.bots,
            fact_checker_ratio=args.fact_checkers,
            seed=args.seed,
        )
        _animation = animate_simulation(
            model,
            steps=args.steps,
            show_graph=not args.no_graph,
            show_timeseries=args.timeseries,
            interval_ms=args.interval,
        )
