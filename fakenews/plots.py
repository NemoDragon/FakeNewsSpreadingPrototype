from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx

from .constants import (
    ROLE_LABELS,
    ROLE_MARKERS,
    ROLE_REGULAR,
    STATE_BELIEVER,
    STATE_INFORMED,
    STATE_SUSCEPTIBLE,
)
from .model import FakeNewsModel
from .results import SimulationResultsRecorder


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
    from matplotlib.lines import Line2D

    color_map = {
        STATE_SUSCEPTIBLE: "#9aa0a6",
        STATE_BELIEVER: "#d93025",
        STATE_INFORMED: "#188038",
    }

    node_states = {agent.pos: agent.state for agent in model.agent_list}
    node_roles = {agent.pos: agent.role for agent in model.agent_list}

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(model.graph, seed=42)
    nx.draw_networkx_edges(model.graph, pos, alpha=0.15, width=0.6)
    for role, marker in ROLE_MARKERS.items():
        nodes = [node for node in model.graph.nodes if node_roles.get(node, ROLE_REGULAR) == role]
        if not nodes:
            continue
        node_colors = [color_map.get(node_states.get(node, STATE_SUSCEPTIBLE)) for node in nodes]
        nx.draw_networkx_nodes(
            model.graph,
            pos,
            nodelist=nodes,
            node_color=node_colors,
            node_shape=marker,
            node_size=60,
        )
    plt.title("Social Graph (colored by state)")
    plt.axis("off")
    role_handles = [
        Line2D([0], [0], marker=marker, color="w", label=ROLE_LABELS[role], markerfacecolor="#444444", markersize=8)
        for role, marker in ROLE_MARKERS.items()
    ]
    state_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Susceptible",
            markerfacecolor=color_map[STATE_SUSCEPTIBLE],
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Believer",
            markerfacecolor=color_map[STATE_BELIEVER],
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Informed",
            markerfacecolor=color_map[STATE_INFORMED],
            markersize=8,
        ),
    ]
    plt.legend(handles=role_handles + state_handles, loc="upper left", fontsize=8)


def animate_simulation(
    model: FakeNewsModel,
    steps: int,
    recorder: Optional[SimulationResultsRecorder] = None,
    show_graph: bool = True,
    show_timeseries: bool = False,
    interval_ms: int = 200,
) -> Tuple[object, "matplotlib.figure.Figure"]:
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
    graph_nodes_by_role = {}
    graph_edges = None

    if ax_graph is not None:
        ax_graph.set_title("Social Graph (colored by state)")
        ax_graph.axis("off")
        graph_edges = nx.draw_networkx_edges(model.graph, pos, ax=ax_graph, alpha=0.15, width=0.6)
        from matplotlib.lines import Line2D

        node_roles = {agent.pos: agent.role for agent in model.agent_list}
        for role, marker in ROLE_MARKERS.items():
            nodes = [node for node in model.graph.nodes if node_roles.get(node, ROLE_REGULAR) == role]
            if not nodes:
                continue
            node_colors = [color_map.get(STATE_SUSCEPTIBLE) for _ in nodes]
            graph_nodes_by_role[role] = nx.draw_networkx_nodes(
                model.graph,
                pos,
                ax=ax_graph,
                nodelist=nodes,
                node_color=node_colors,
                node_shape=marker,
                node_size=40,
            )

        role_handles = [
            Line2D([0], [0], marker=marker, color="w", label=ROLE_LABELS[role], markerfacecolor="#444444", markersize=8)
            for role, marker in ROLE_MARKERS.items()
        ]
        state_handles = [
            Line2D([0], [0], marker="o", color="w", label="Susceptible", markerfacecolor=color_map[STATE_SUSCEPTIBLE], markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Believer", markerfacecolor=color_map[STATE_BELIEVER], markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Informed", markerfacecolor=color_map[STATE_INFORMED], markersize=8),
        ]
        ax_graph.legend(handles=role_handles + state_handles, loc="upper left", fontsize=7)

    def update(time_step: int):
        model.step()
        if recorder is not None:
            recorder.record_step(time_step, model)

        ts_susceptible.append(model.count_state(STATE_SUSCEPTIBLE))
        ts_believer.append(model.count_state(STATE_BELIEVER))
        ts_informed.append(model.count_state(STATE_INFORMED))

        artists: List[object] = []
        if ax_ts is not None and line_sus is not None and line_bel is not None and line_inf is not None:
            x = list(range(len(ts_susceptible)))
            line_sus.set_data(x, ts_susceptible)
            line_bel.set_data(x, ts_believer)
            line_inf.set_data(x, ts_informed)
            ax_ts.set_xlim(0, max(10, len(ts_susceptible)))
            ax_ts.set_ylim(0, model.num_agents)
            artists.extend([line_sus, line_bel, line_inf])

        if graph_nodes_by_role:
            node_states = {agent.pos: agent.state for agent in model.agent_list}
            node_roles = {agent.pos: agent.role for agent in model.agent_list}
            for role, collection in graph_nodes_by_role.items():
                nodes = [node for node in model.graph.nodes if node_roles.get(node, ROLE_REGULAR) == role]
                node_colors = [color_map.get(node_states.get(node, STATE_SUSCEPTIBLE)) for node in nodes]
                collection.set_color(node_colors)
                artists.append(collection)
            if graph_edges is not None:
                artists.append(graph_edges)
            if ax_graph is not None:
                ax_graph.set_title(
                    f"Step {time_step} | S:{ts_susceptible[-1]} B:{ts_believer[-1]} I:{ts_informed[-1]}"
                )

        return tuple(artists)

    # Do one step immediately so the window does not look frozen.
    if steps >= 1:
        update(1)
        frame_iter = range(2, steps + 1)
    else:
        frame_iter = []

    animation = FuncAnimation(fig, update, frames=frame_iter, interval=interval_ms, blit=False, repeat=False)
    plt.tight_layout()
    return animation, fig
