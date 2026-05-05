# Fake News Simulation - Documentation

This document explains how the prototype in `fake_news_sim.py` works, what each class and function does, and what parameter patterns are used.

## 1. Overview

The script simulates fake news diffusion on a scale-free social network built with the Barabasi-Albert model. Agents have roles, states, and behavioral parameters. During each step, agents influence each other through their neighbors, and the simulation can be shown as:

- a static time series chart,
- a static graph view,
- an animation where the graph updates step by step.

The model uses:

- `networkx` for graph generation,
- `mesa` for agent-based modeling primitives,
- `matplotlib` for plots and animation.

## 2. Constants and role patterns

The code defines two main groups of constants.

### 2.1 State constants

- `STATE_SUSCEPTIBLE = "susceptible"`
- `STATE_BELIEVER = "believer"`
- `STATE_INFORMED = "informed"`

These represent the SIR-like state of an agent.

### 2.2 Role constants

- `ROLE_REGULAR = "regular"`
- `ROLE_INFLUENCER = "influencer"`
- `ROLE_FACT_CHECKER = "fact_checker"`
- `ROLE_BOT = "bot"`

These describe the agent type and determine visual shape and behavior.

### 2.3 Visualization mappings

- `ROLE_MARKERS` maps role to marker shape:
  - regular user -> `o`
  - influencer -> `^`
  - fact-checker -> `s`
  - bot -> `X`
- `ROLE_LABELS` maps role to a readable legend label.

## 3. Data container

### `AgentParams`

`AgentParams` is a `dataclass` holding the parameters of one agent.

Fields:

- `belief: float`
- `competence: float`
- `stubbornness: float`
- `fact_check_prob: float`
- `forgetting_prob: float`
- `engagement: float`
- `trust: Dict[str, float]`
- `homophily_threshold: float`

### Parameter pattern used here

Most parameters are sampled from small ranges using `random.uniform(...)`. This makes the prototype heterogeneous instead of assigning the same values to every agent.

Typical ranges used in the current code:

- `belief`: `0.4 - 0.6`
- `competence`: `0.2 - 0.8`
- `stubbornness`: `0.1 - 0.7`
- `fact_check_prob`: `0.05 - 0.3`
- `forgetting_prob`: `0.01 - 0.05`
- `engagement`: `0.2 - 0.9`
- `homophily_threshold`: `0.1 - 0.6`

These values are sampled from a uniform distribution:

$$
x \sim \mathcal{U}(a, b)
$$

where $a$ and $b$ are the lower and upper bounds of the chosen range.

The `trust` field is a dictionary indexed by source role:

- trust in regular users,
- trust in influencers,
- trust in fact-checkers,
- trust in bots.

## 4. Class: `SocialAgent`

`SocialAgent` is the basic agent in the model. It inherits from `mesa.Agent` and stores:

- `unique_id`
- `role`
- `state`
- `params`

### Constructor

#### `__init__(unique_id, model, role, state, params)`

What it does:

- initializes the Mesa base class,
- stores the agent id,
- stores the role and state,
- stores the parameter bundle.

Parameter pattern:

- `unique_id: int`
- `model: Model`
- `role: str`
- `state: str`
- `params: AgentParams`

### `step()`

This is the main behavior entry point for one simulation tick.

Behavior pattern:

- if the agent is a bot, call `_bot_step()`,
- if the agent is a fact-checker, call `_fact_checker_step()`,
- otherwise call `_interact_with_neighbors()` and then `_update_state()`.

This means bots and fact-checkers have specialized behavior, while regular users and influencers use the general interaction logic.

### `_bot_step()`

Purpose:

- bots amplify misinformation.

Behavior pattern:

- generate a random number,
- if it is greater than `model.bot_post_prob`, do nothing,
- otherwise broadcast positive influence with `model.bot_influence`.

Parameter pattern:

- no external parameters,
- uses model-level constants.

### `_fact_checker_step()`

Purpose:

- fact-checkers push corrective information.

Behavior pattern:

- broadcast negative influence with `model.fact_checker_influence`.

Parameter pattern:

- no external parameters,
- uses model-level constants.

### `_interact_with_neighbors()`

Purpose:

- regular users and influencers interact with sampled neighbors.

Behavior pattern:

- get neighbors from the graph,
- decide how many neighbors to sample based on engagement,
- skip interactions if belief distance is too large,
- if the neighbor is a believer or bot, apply positive influence,
- if the neighbor is informed or fact-checker, apply negative influence.

Parameter pattern:

- `engagement` controls how many neighbors are sampled,
- `homophily_threshold` blocks interactions between agents whose beliefs are too far apart.

Mathematical pattern:

$$
k = \max\left(1, \operatorname{round}(e \cdot d)\right)
$$

where:

- $e$ is engagement,
- $d$ is the number of neighbors,
- $k$ is the number of sampled neighbors.

The homophily rule is:

$$
|b_i - b_j| \leq h
$$

where:

- $b_i$ is the belief of the current agent,
- $b_j$ is the belief of the neighbor,
- $h$ is the homophily threshold.

### `_broadcast_influence(base_influence, source_role)`

Purpose:

- push an influence value to all neighbors.

Parameter pattern:

- `base_influence: float` is the raw effect of the message,
- `source_role: str` selects which trust value should be used.

Behavior pattern:

- iterate over all neighbors,
- skip if belief distance is greater than `homophily_threshold`,
- call `_apply_influence(...)` on the neighbor.

The same interaction condition applies here:

$$
|b_i - b_j| \leq h
$$

### `_apply_influence(base_influence, source_role)`

Purpose:

- update the target agent belief after receiving a message.

Parameter pattern:

- `base_influence: float` can be positive or negative,
- `source_role: str` selects trust from the `trust` dictionary.

Mathematical pattern:

$$
t = T(r)
$$

$$
r_s = 1 - s
$$

If the influence is positive, competence reduces the effect further:

$$
r_+ = (1 - s)(1 - c)
$$

The belief update is:

$$
b' = \operatorname{clip}_{[0,1]}\left(b + \Delta \cdot t \cdot r\right)
$$

where:

- $b$ is the current belief,
- $b'$ is the updated belief,
- $\Delta$ is `base_influence`,
- $t$ is trust for the source role,
- $s$ is stubbornness,
- $c$ is competence,
- $\operatorname{clip}_{[0,1]}$ clamps the result to the interval $[0,1]$.

Behavior pattern:

- read trust for the source role,
- compute resistance as `1.0 - stubbornness`,
- if the influence is positive, reduce resistance further using competence,
- update belief and clamp it to `[0.0, 1.0]`,
- if the influence is negative, there is an additional fact-checking chance controlled by `fact_check_prob`.

### `_update_state()`

Purpose:

- convert belief values into state transitions.

Behavior pattern:

- if the agent is believer or informed, it may forget and return to susceptible,
- if susceptible and belief is high enough, switch to believer,
- if susceptible and belief is low enough, switch to informed,
- if believer and belief drops low enough, switch to informed,
- if informed and belief rises high enough, switch to believer.

Threshold pattern used here:

- upper threshold: `model.belief_upper`
- lower threshold: `model.belief_lower`

State transitions can be written as:

$$
	ext{S} \to \text{B} \quad \text{if} \quad b \geq \theta_u
$$

$$
	ext{S} \to \text{I} \quad \text{if} \quad b \leq \theta_l
$$

$$
	ext{B} \to \text{I} \quad \text{if} \quad b \leq \theta_l
$$

$$
	ext{I} \to \text{B} \quad \text{if} \quad b \geq \theta_u
$$

where $\theta_u$ is the upper threshold and $\theta_l$ is the lower threshold.

## 5. Class: `FakeNewsModel`

`FakeNewsModel` is the simulation container. It builds the graph, creates agents, stores global parameters, and counts states.

### Constructor

#### `__init__(n_agents=200, m_links=3, influencer_ratio=0.05, bot_ratio=0.03, fact_checker_ratio=0.03, seed=42)`

What it does:

- sets the random seed,
- creates a Barabasi-Albert graph,
- creates the NetworkX grid wrapper,
- creates and stores all agents,
- defines global influence constants,
- prepares the data collector.

Parameter pattern:

- `n_agents`: size of the network,
- `m_links`: number of edges attached when a new node joins the BA graph,
- `influencer_ratio`: share of highest-degree nodes marked as influencers,
- `bot_ratio`: share of nodes assigned as bots,
- `fact_checker_ratio`: share of nodes assigned as fact-checkers,
- `seed`: controls reproducibility.

### `_init_agents(influencer_ratio, bot_ratio, fact_checker_ratio)`

Purpose:

- assign roles and initial states to all nodes.

Behavior pattern:

- sort graph nodes by degree,
- choose top-degree nodes as influencers,
- choose random remaining nodes as bots and fact-checkers,
- create `SocialAgent` objects,
- place each agent on the graph node.

Role and state pattern used:

- regular user -> `susceptible`
- influencer -> `susceptible`
- bot -> `believer`
- fact-checker -> `informed`

### `_sample_params(role)`

Purpose:

- generate a heterogeneous parameter set for one agent.

Parameter pattern:

- `role: str` is used only to slightly modify engagement for influencers and bots.

Behavior pattern:

- generate random values for cognitive and social properties,
- boost influencer engagement,
- set bot engagement to maximum,
- create role trust dictionary,
- return an `AgentParams` object.

Mathematical pattern:

$$
e' =
\begin{cases}
\min(1, e + 0.2) & \text{for influencers} \\
1 & \text{for bots} \\
e & \text{otherwise}
\end{cases}
$$

For trust values, the code uses role-dependent uniform sampling:

$$
T(r) \sim \mathcal{U}(a_r, b_r)
$$

### `_count_state(state)`

Purpose:

- count how many agents are currently in one state.

Parameter pattern:

- `state: str` is one of the three state constants.

### `step()`

Purpose:

- perform one simulation tick.

Behavior pattern:

- collect model metrics,
- shuffle the agent list,
- call `step()` on each agent.

This gives a simple discrete-time update loop.

## 6. Free functions

### `run_simulation(steps=50, n_agents=200, m_links=3, influencer_ratio=0.05, bot_ratio=0.03, fact_checker_ratio=0.03, seed=42)`

Purpose:

- run the model for a fixed number of steps and return results.

Parameter pattern:

- `steps: int` controls how many ticks are simulated,
- the remaining arguments are passed directly to `FakeNewsModel`.

Return pattern:

- returns a tuple:
  - the model instance,
  - the susceptible count list,
  - the believer count list,
  - the informed count list.

### `plot_timeseries(susceptible, believer, informed)`

Purpose:

- draw the time series chart.

Parameter pattern:

- `susceptible: List[int]`
- `believer: List[int]`
- `informed: List[int]`

Behavior pattern:

- create a figure,
- plot three lines,
- add title, axes labels, legend, and layout.

### `plot_graph(model)`

Purpose:

- draw a static graph view with role shapes and state colors.

Parameter pattern:

- `model: FakeNewsModel`

Behavior pattern:

- compute spring layout positions,
- draw edges,
- draw nodes in groups by role using different markers,
- color nodes by current state,
- add a legend for both roles and states.

### `node_size_from_degree(degree)`

Purpose:

- convert a node degree into a display size.

Parameter pattern:

- `degree: int`

Behavior pattern:

- larger degree means larger displayed node,
- formula used: `40 + (degree ** 1.25) * 12`.

### `node_sizes_for_graph(graph)`

Purpose:

- precompute display sizes for all nodes.

Parameter pattern:

- `graph: nx.Graph`

Return pattern:

- dictionary mapping node id to node size.

### `animate_simulation(model, steps, show_graph=True, show_timeseries=False, interval_ms=200)`

Purpose:

- run a visible animation where the graph updates step by step.

Parameter pattern:

- `model: FakeNewsModel` is the simulation state,
- `steps: int` is the number of animation frames,
- `show_graph: bool` toggles graph display,
- `show_timeseries: bool` toggles a second panel with the line chart,
- `interval_ms: int` controls frame delay in milliseconds.

Behavior pattern:

- create one or two matplotlib axes depending on options,
- prepare line objects for the time series if needed,
- prepare graph node collections grouped by role,
- define an inner `update(frame_index)` function,
- on each frame, call `model.step()` and refresh colors and counts,
- create `FuncAnimation`, render the first frame immediately, and display the window.

Return pattern:

- returns the animation object so it is not garbage-collected too early.

## 7. Command-line interface

The script uses `argparse` in the `__main__` block.

### CLI arguments and patterns

- `--agents` -> number of agents, default `200`
- `--steps` -> number of simulation steps, default `60`
- `--m-links` -> BA graph attachment parameter, default `3`
- `--influencers` -> influencer ratio, default `0.05`
- `--bots` -> bot ratio, default `0.03`
- `--fact-checkers` -> fact-checker ratio, default `0.03`
- `--seed` -> random seed, default `42`
- `--no-graph` -> disables graph visualization
- `--no-animate` -> disables animation and shows static plots
- `--interval` -> animation frame interval in ms, default `200`
- `--timeseries` -> shows time series next to the graph in animation mode

### Main execution pattern

There are two execution branches:

1. If `--no-animate` is used:
   - run the full simulation first,
   - show the static time series,
   - optionally show the static graph.

2. If animation is enabled:
   - create the model,
   - run `animate_simulation(...)`,
   - keep the returned animation object alive in `_animation`.

## 8. How the parameters work together

### Cognitive parameters

- `belief` controls the current conviction level.
- `competence` weakens positive misinformation influence.
- `stubbornness` weakens all influence effects.
- `fact_check_prob` controls correction-triggered self-verification.
- `forgetting_prob` enables loss of attention and returns to susceptible.

### Social parameters

- `engagement` controls how many neighbors are sampled each step.
- `trust` controls how much a message from a given role affects belief.
- `homophily_threshold` blocks interactions between very dissimilar agents.

### Model parameters

- `n_agents` controls graph size.
- `m_links` controls graph density.
- `influencer_ratio`, `bot_ratio`, `fact_checker_ratio` control role distribution.
- `seed` makes the network and sampling reproducible.

### Visual parameters

- role -> shape,
- state -> color,
- degree -> size.

That combination makes the graph readable:

- shape shows what type of agent it is,
- color shows what state it is in,
- size shows how connected it is.

## 9. Short example usage

Run animated simulation with default settings:

```bash
python fake_news_sim.py
```

Run animation with a time series panel:

```bash
python fake_news_sim.py --timeseries
```

Run a larger simulation:

```bash
python fake_news_sim.py --agents 400 --steps 120 --m-links 4
```

Run the static version only:

```bash
python fake_news_sim.py --no-animate
```

## 10. Summary

This prototype uses a simple but clear pattern:

- graph topology from Barabasi-Albert,
- roles assigned by node structure and random sampling,
- belief-based state transitions,
- role shapes, state colors, and degree-based sizes in the visualization.

It is intentionally lightweight and suitable as a proof of concept rather than a fully optimized simulator.
