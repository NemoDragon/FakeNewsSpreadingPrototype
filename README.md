# Fake News Spreading Prototype

An agent-based simulation built with **Mesa** and **NetworkX**, modeling how misinformation spreads through a **Barabási–Albert** social graph populated with four agent roles - regular users, influencers, bots and fact-checkers - each governed by belief dynamics (trust, stubbornness, competence, homophily). Belief can optionally be nudged by one or more **TF-IDF + Logistic Regression** classifiers - either a single shared classifier or an independently trained one per agent - trained on the [Fake or Real News](https://github.com/GeorgeMcIntire/fake_real_news_dataset) dataset, which predicts whether the news item an agent is holding is `FAKE` or `REAL`, drives belief updates and decides whether the agent forwards that item to its neighbors. The simulation runs live as a **Matplotlib** animation (social graph + state time series) and every step, starting from the initial configuration, is logged to disk for later analysis.

## Table of contents

- [How it works](#how-it-works)
- [Project architecture](#project-architecture)
- [Agent roles and belief dynamics](#agent-roles-and-belief-dynamics)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the simulation](#running-the-simulation)
- [CLI arguments](#cli-arguments)
- [Output files](#output-files)
- [Experiment results](#experiment-results)
- [Known limitations](#known-limitations)
- [Repository structure](#repository-structure)
- [Possible improvements](#possible-improvements)

## How it works

Each run goes through the following stages:

1. **Dataset loading** - if a news CSV is given (`--news-csv`, default `fake_or_real_news.csv`), `FakeNewsDataset.from_csv` reads it into `NewsItem`s (title, text, `FAKE`/`REAL` label) via `FakeOrRealNewsCSVReader`, skipping rows with an empty text or an unrecognized label.
2. **Graph construction** - `FakeNewsModel` builds a `networkx.barabasi_albert_graph(n_agents, m_links, seed=seed)` and wraps it in a Mesa `NetworkGrid`.
3. **Classifier training** - the dataset is shuffled and split 90/10 into train/test (`_split_dataset`). If `--train-samples-per-agent` is `0`, a single `FakeNewsClassifier` (`TfidfVectorizer` + `LogisticRegression(class_weight="balanced")`) is fit on the whole training split and shared by every agent; otherwise each agent instead gets its own classifier trained on an independent, class-balanced chunk of that many examples drawn from the training split (see [Agent roles and belief dynamics](#agent-roles-and-belief-dynamics)). The test split supplies the news items agents actually carry during the run.
4. **Agent initialization** (`_init_agents`) - nodes are ranked by graph degree; the top `influencer_ratio` become **influencers**. Of the rest, `bot_ratio` become **bots** (each seeded with a random `FAKE` item from the test split, always starting `believer`) and `fact_checker_ratio` become **fact-checkers** (seeded with a random `REAL` item, always starting `informed`); everyone else is a **regular** user. Regular users and influencers instead draw their initial state from `--susceptible`/`--believer`/`--informed`, three ratios that are normalized to sum to 1.0 (`_normalize_state_distribution`) and sampled per agent (`_pick_initial_state`). Every agent gets randomly sampled `AgentParams` (belief, competence, stubbornness, fact-check probability, forgetting probability, engagement, per-role trust, homophily threshold).
5. **Stepping** - on each `model.step()`, agents are shuffled and each one acts: regular/influencer agents sample a fraction of their neighbors (set by `engagement`), update belief based on what those neighbors are spreading, and forward any news item their own classifier judged `REAL` to their own neighbors; bots broadcast misinformation to all neighbors with probability `bot_post_prob` (`1` by default - i.e. every step); fact-checkers broadcast corrections to all neighbors every step (see [Agent roles and belief dynamics](#agent-roles-and-belief-dynamics)).
6. **Visualization and recording** - `animate_simulation` records and renders the initial state (step 0) before the stepping loop starts, then redraws the social graph (colored by state, shaped by role) and/or a `susceptible`/`believer`/`informed` time series every frame; `SimulationResultsRecorder` records the same per-step counts and the full per-agent state into memory, then writes them to disk once the run ends.

## Project architecture

| Module | Responsibility |
|---|---|
| `fake_news_sim.py` | CLI entry point: parses arguments, builds the dataset/model/recorder, runs the simulation (animated or static) and triggers plotting |
| `fakenews/model.py` | `FakeNewsModel`: builds the Barabási–Albert graph, trains the shared or per-agent classifier(s), normalizes the initial-state distribution, assigns roles/states/params to agents, and steps the population each tick |
| `fakenews/agents.py` | `SocialAgent` and `AgentParams`: per-role step logic (regular/influencer interaction + forwarding, bot broadcast, fact-checker broadcast) and the belief-update/state-transition rules |
| `fakenews/news.py` | `NewsItem`, `FakeNewsDataset`, `FakeOrRealNewsCSVReader`, `FakeNewsClassifier` (TF-IDF + Logistic Regression) and `predict_news`, the classifier-inference helper agents call into |
| `fakenews/results.py` | `SimulationResultsRecorder`: accumulates per-step state counts and per-agent rows, then writes the `parameters.json` / `timeseries.csv` / `graph.csv` output files |
| `fakenews/plots.py` | `plot_timeseries`, `plot_graph` (static Matplotlib views) and `animate_simulation` (live `FuncAnimation` combining both, wired to the recorder, starting from the recorded step-0 state) |
| `fakenews/constants.py` | Shared state (`susceptible`/`believer`/`informed`) and role (`regular`/`influencer`/`fact_checker`/`bot`) string constants, plus their plot markers/labels |
| `fake_or_real_news.csv` / `.xlsx` | The bundled [Fake or Real News](https://github.com/GeorgeMcIntire/fake_real_news_dataset) dataset (6,335 labeled articles) used both to train the classifier(s) and to seed bots/fact-checkers with content |
| `test_scenarios.txt` / `scenariusze_testowe.txt` | The 15 reproducible CLI commands behind [Experiment results](#experiment-results), in English and Polish |

Data flows one-directionally per step: neighbor interactions and role broadcasts update an agent's `belief` (`AgentParams.belief`) and, if a news item is being adopted, the relevant classifier's prediction on that item nudges belief further and decides whether the item gets forwarded next step -> the updated `belief` is thresholded into a `susceptible`/`believer`/`informed` state -> both feed the live plots and the recorder's per-step rows.

## Agent roles and belief dynamics

| Role | Starting state | Per-step behavior |
|---|---|---|
| `regular` | drawn from `--susceptible`/`--believer`/`--informed` | Samples `engagement` |neighbors|` neighbors; adopts belief pressure from any that are `believer`/bots (+) or `informed`/fact-checkers (-); forwards any pending news item afterward |
| `influencer` | drawn from `--susceptible`/`--believer`/`--informed` | Same as regular, but sampled with a boosted `engagement` (+0.2, capped at 1.0), so it interacts with more neighbors per step |
| `bot` | always `believer` | Never changes its own state; with probability `bot_post_prob` (`1` by default) it broadcasts `+bot_influence` to every neighbor at once, along with its seeded `FAKE` news item |
| `fact_checker` | always `informed` | Never changes its own state; every step it broadcasts `-fact_checker_influence` to every neighbor, along with its seeded `REAL` news item |

Belief updates (`SocialAgent._apply_influence`) are shaped by the receiving agent's own parameters: `trust` in the sender's role scales the raw influence, `stubbornness` provides general resistance, and `competence` adds extra resistance specifically against positive (misinformation-favoring) influence. Interactions are further gated by **homophily** - two agents only influence each other if their belief values differ by less than the receiver's `homophily_threshold`. When exposed to a correction (`base_influence < 0`), an agent also has a `fact_check_prob` chance of an extra, independent belief drop (self-verification).

**Classifier-assisted belief and forwarding.** Whenever an agent receives a news item through an interaction, it asks a classifier - its own, if `--train-samples-per-agent > 0`, otherwise the model's shared one - for a `FAKE`/`REAL` prediction and a confidence (`_apply_classifier_influence`). Two independent effects follow:

- **Belief**, if `--classifier-weight > 0`: belief is nudged toward or away from the prediction, scaled by `classifier_weight` and the prediction's confidence - correct predictions pull belief toward the truth, incorrect ones push it further from it.
- **Forwarding**, regardless of `classifier_weight`: a regular user or influencer queues the item to forward to its own neighbors next step (`_pending_share_item`) if the item was predicted `REAL`; a correctly-identified `FAKE` item is not forwarded. Bots and fact-checkers are unaffected - they always broadcast their seeded item.

States transition on fixed thresholds (`belief_upper = 0.7`, `belief_lower = 0.3`) plus a small `forgetting_prob` chance per step of decaying back to `susceptible` from either `believer` or `informed`.

## Requirements

- Python **3.10+**
- [`mesa`](https://mesa.readthedocs.io/) - agent-based modeling framework (`Agent`, `Model`, `NetworkGrid`)
- `networkx` - Barabási–Albert graph generation
- `matplotlib` - static plots and `FuncAnimation`
- `scikit-learn` - `TfidfVectorizer`, `LogisticRegression`, `Pipeline`

No `requirements.txt` is provided; install the packages directly (see [Installation](#installation)).

## Installation

```bash
git clone https://github.com/NemoDragon/FakeNewsSpreadingPrototype.git
cd FakeNewsSpreadingPrototype

python3 -m venv venv
source venv/bin/activate   # linux/macOS
venv\Scripts\activate      # windows

pip install mesa networkx matplotlib scikit-learn
```

The bundled `fake_or_real_news.csv` is used automatically as the default news dataset - no separate download is needed.

## Running the simulation

```bash
# default run: 200 agents, 60 steps, live animation, classifier disabled (weight 0)
python fake_news_sim.py

# smaller, faster run with the shared classifier nudging belief
python fake_news_sim.py --agents 80 --steps 40 --classifier-weight 0.3

# per-agent classifiers, each trained on 5 examples, instead of one shared classifier
python fake_news_sim.py --classifier-weight 0.5 --train-samples-per-agent 5

# static plots instead of an animation, with both graph and time series
python fake_news_sim.py --no-animate --timeseries

# start from an already-polarized population (50% believers, 40% susceptible, 10% informed)
python fake_news_sim.py --susceptible 0.40 --believer 0.50 --informed 0.10 --classifier-weight 1.0
```

## CLI arguments

| Argument | Default | Meaning |
|---|---|---|
| `--agents` | `200` | Number of agents (graph nodes) |
| `--steps` | `60` | Number of simulation steps |
| `--m-links` | `3` | Barabási–Albert edges attached per new node |
| `--influencers` | `0.05` | Fraction of highest-degree nodes made influencers |
| `--bots` | `0.03` | Fraction of remaining nodes made bots |
| `--fact-checkers` | `0.03` | Fraction of remaining nodes made fact-checkers |
| `--susceptible` | `1.0` | Initial `susceptible` share for regular users and influencers (normalized with the two below) |
| `--believer` | `0.0` | Initial `believer` share for regular users and influencers |
| `--informed` | `0.0` | Initial `informed` share for regular users and influencers |
| `--seed` | `42` | Random seed (graph generation + `random` module) |
| `--no-graph` | off | Disable the social-graph visualization |
| `--no-animate` | off | Disable the live animation, run headless and show static plots instead |
| `--interval` | `200` | Animation frame interval, in milliseconds |
| `--timeseries` | off | Show the state time series alongside the graph |
| `--news-csv` | `fake_or_real_news.csv` | CSV of labeled news items used to train the classifier(s) and seed bots/fact-checkers |
| `--output-dir` | `results` | Directory the recorder writes its output files to |
| `--classifier-weight` | `0.0` | Strength of the classifier's nudge on belief (`0` disables belief influence; forwarding is unaffected) |
| `--train-samples-per-agent` | `10` | If `> 0`, train a separate, class-balanced classifier per agent on this many examples instead of one shared classifier |

## Output files

Each run creates three timestamped files (`YYYYMMDD_HHMMSS_*`) under `--output-dir` via `SimulationResultsRecorder`, starting with the initial state at `time_step = 0`:

| File | Contents |
|---|---|
| `*_parameters.json` | The run's configuration - agent counts, ratios, initial-state distribution, seed, classifier settings |
| `*_timeseries.csv` | One row per step (including step 0): `time_step`, `susceptible`, `believer`, `informed` counts |
| `*_graph.csv` | One row per agent per step (`;`-delimited): role, state, belief and every `AgentParams` field, the held news item's id/title/text/true label, the classifier's predicted label/confidence for it, and whether the shared (`model_classifier_trained`) and/or that agent's own (`agent_classifier_trained`) classifier was successfully trained |

## Experiment results

A 15-scenario sweep (commands in `test_scenarios.txt`, all with `--timeseries`, agent/step defaults unless noted) explores how the classifier, network topology, agent-role ratios and initial-state distribution shape the step-60 outcome:

| # | Scenario | Key variable | Believers | Informed | Informed:Believer |
|---|---|---|---|---|---|
| 0 | Baseline, no classifier | `classifier-weight=0.0` | 178 | 17 | 0.10 |
| 1 | Baseline, classifier on | `classifier-weight=1.0` | 54 | 139 | 2.57 |
| 2 | Small network (step 40) | `agents=60, m-links=2` | 7 | 34 | 4.86 |
| 3 | Large network (step 80) | `agents=500, steps=80` | 138 | 348 | 2.52 |
| 4 | Sparse graph | `m-links=1` | 38 | 41 | 1.08 |
| 5 | Dense graph | `m-links=8` | 33 | 167 | 5.06 |
| 6 | Many influencers | `influencers=0.20` | 48 | 144 | 3.00 |
| 7 | No influencers | `influencers=0.0` | 98 | 96 | 0.98 |
| 8 | Many bots | `bots=0.15` | 89 | 104 | 1.17 |
| 9 | No bots | `bots=0.0` | 49 | 143 | 2.92 |
| 10 | Many fact-checkers | `fact-checkers=0.15` | 34 | 150 | 4.41 |
| 11 | No fact-checkers | `fact-checkers=0.0` | 59 | 119 | 2.02 |
| 12 | Mostly susceptible at start | `susceptible=0.95` | 74 | 116 | 1.57 |
| 13 | Many believers at start | `believer=0.50` | 69 | 123 | 1.78 |
| 14 | Many informed at start | `informed=0.50` | 54 | 141 | 2.61 |

Headline findings from the accompanying project report:

- **The classifier is the single biggest lever tested**: enabling it (Experiment 1 vs. 0) turns an 8× believer-dominated outcome (178 believers, 17 informed) into an informed-dominated one (54 believers, 139 informed) with no other change.
- **Network density is a strong, nonlinear regulator**: dropping `m-links` from 3 to 1 (Experiment 4) cuts informed agents by 71%, while raising it to 8 (Experiment 5) fully exhausts the susceptible pool and pushes the informed:believer ratio to 5.06:1, the highest observed.
- **Influencers and fact-checkers act asymmetrically**: removing influencers entirely (Experiment 7) costs 31% of the informed population, versus only a 14% loss from removing fact-checkers (Experiment 11) - the classifier picks up much of the fact-checkers' corrective role on its own, but there is no equivalent stand-in for influencer reach.
- **The equilibrium is fairly robust to initial conditions**: starting from 95% susceptible, 50% believers, or 50% informed (Experiments 12-14) all converge to informed:believer ratios within roughly 1.6-2.6, a much narrower spread than the topology and role-ratio experiments produce.

## Known limitations

- The commented-out branch in `_apply_classifier_influence` (`# if direction < 0 and self.params.belief > random.random() * confidence * self.params.competence * 4: direction *= -1.0`) was an earlier "resistance to correction" mechanic that is currently disabled dead code; it can be removed or reinstated depending on which belief dynamics the next iteration should use.
- `FakeNewsClassifier.train` silently falls back to an untrained state (`is_trained = False`, all predictions default to 0.5) if a training split - the shared one, or an individual agent's chunk under `--train-samples-per-agent` - ends up with only one class present, which is more likely for small per-agent chunks than for the shared classifier.
- With `--train-samples-per-agent > 0`, every agent gets its own classifier regardless of role, including bots and fact-checkers, even though only regular users and influencers currently use it for forwarding decisions and only agents that receive a news item during an interaction use it for belief at all.

## Repository structure

```
FakeNewsSpreadingPrototype/
├── fakenews/
│   ├── __init__.py         # Re-exports state/role constants
│   ├── constants.py        # State/role string constants + plot markers/labels
│   ├── agents.py           # SocialAgent, AgentParams: per-role step logic, belief updates, forwarding
│   ├── model.py             # FakeNewsModel: graph, classifier(s) training, agent init, stepping
│   ├── news.py              # NewsItem, FakeNewsDataset, CSV reader, FakeNewsClassifier
│   ├── plots.py              # Static plots + live FuncAnimation (including step-0 rendering)
│   └── results.py            # SimulationResultsRecorder: JSON/CSV output
├── fake_news_sim.py          # CLI entry point
├── fake_or_real_news.csv     # Bundled dataset (6,335 labeled news articles)
├── fake_or_real_news.xlsx    # Same dataset, spreadsheet format
├── test_scenarios.txt        # 15 reproducible experiment commands (English)
└── scenariusze_testowe.txt   # Same 15 experiment commands (Polish)
```
