"""Run the fake news diffusion simulation (thin entry point)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

from fakenews.model import FakeNewsModel
from fakenews.news import FakeNewsDataset
from fakenews.plots import animate_simulation, plot_graph, plot_timeseries
from fakenews.results import SimulationResultsRecorder


def run_simulation(
    steps: int = 50,
    n_agents: int = 200,
    m_links: int = 3,
    influencer_ratio: float = 0.05,
    bot_ratio: float = 0.03,
    fact_checker_ratio: float = 0.03,
    susceptible_ratio: float = 1.0,
    believer_ratio: float = 0.0,
    informed_ratio: float = 0.0,
    seed: int = 42,
    news_csv: str | None = None,
    output_dir: str | Path = "results",
    classifier_weight: float = 0.0,
    train_samples_per_agent: int = 0,
    animate: bool = True,
    show_graph: bool = True,
    show_timeseries: bool = False,
    interval_ms: int = 200,
) -> Tuple[FakeNewsModel, List[int], List[int], List[int], SimulationResultsRecorder]:
    dataset = FakeNewsDataset.from_csv(news_csv) if news_csv else None

    model = FakeNewsModel(
        n_agents=n_agents,
        m_links=m_links,
        influencer_ratio=influencer_ratio,
        bot_ratio=bot_ratio,
        fact_checker_ratio=fact_checker_ratio,
        susceptible_ratio=susceptible_ratio,
        believer_ratio=believer_ratio,
        informed_ratio=informed_ratio,
        seed=seed,
        news_dataset=dataset,
        train_classifier=True,
        classifier_weight=classifier_weight,
        train_samples_per_agent=train_samples_per_agent,
    )

    recorder = SimulationResultsRecorder(output_dir=output_dir)
    recorder.set_parameters(
        {
            "steps": steps,
            "n_agents": n_agents,
            "m_links": m_links,
            "influencer_ratio": influencer_ratio,
            "bot_ratio": bot_ratio,
            "fact_checker_ratio": fact_checker_ratio,
            "susceptible_ratio": susceptible_ratio,
            "believer_ratio": believer_ratio,
            "informed_ratio": informed_ratio,
            "seed": seed,
            "news_csv": str(news_csv) if news_csv else None,
            "animate": animate,
            "show_graph": show_graph,
            "show_timeseries": show_timeseries,
            "interval_ms": interval_ms,
            "classifier_weight": classifier_weight,
            "train_samples_per_agent": train_samples_per_agent,
        }
    )

    if animate:
        _animation, _fig = animate_simulation(
            model,
            steps=steps,
            recorder=recorder,
            show_graph=show_graph,
            show_timeseries=show_timeseries,
            interval_ms=interval_ms,
        )
        plt.show()
    else:
        susceptible: List[int] = []
        believer: List[int] = []
        informed: List[int] = []
        recorder.record_step(0, model)
        susceptible.append(model.count_state("susceptible"))
        believer.append(model.count_state("believer"))
        informed.append(model.count_state("informed"))
        for step in range(1, steps + 1):
            model.step()
            recorder.record_step(step, model)
            susceptible.append(model.count_state("susceptible"))
            believer.append(model.count_state("believer"))
            informed.append(model.count_state("informed"))

        plot_timeseries(susceptible, believer, informed)
        if show_graph:
            plot_graph(model)
        plt.show()

    recorder.write_files()

    # For the non-animated path, return the last in-memory timeseries too.
    if animate:
        return model, [], [], [], recorder
    return model, susceptible, believer, informed, recorder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake news spreading")
    parser.add_argument("--agents", type=int, default=200, help="Number of agents")
    parser.add_argument("--steps", type=int, default=60, help="Number of simulation steps")
    parser.add_argument("--m-links", type=int, default=3, help="BA model links per new node")
    parser.add_argument("--influencers", type=float, default=0.05, help="Influencer ratio")
    parser.add_argument("--bots", type=float, default=0.03, help="Bot ratio")
    parser.add_argument("--fact-checkers", type=float, default=0.03, help="Fact-checker ratio")
    parser.add_argument("--susceptible", type=float, default=1.0, help="Initial susceptible share for regular users and influencers")
    parser.add_argument("--believer", type=float, default=0.0, help="Initial believer share for regular users and influencers")
    parser.add_argument("--informed", type=float, default=0.0, help="Initial informed share for regular users and influencers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph visualization")
    parser.add_argument("--no-animate", action="store_true", help="Disable animation and show static plots")
    parser.add_argument("--interval", type=int, default=200, help="Animation frame interval in ms")
    parser.add_argument("--timeseries", action="store_true", help="Show time series alongside graph")
    parser.add_argument("--news-csv", type=str, default="fake_or_real_news.csv", help="CSV with fake/real news")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for result files")
    parser.add_argument(
        "--classifier-weight",
        type=float,
        default=0.0,
        help="How strongly classifier nudges belief (0 disables influence)",
    )
    parser.add_argument(
        "--train-samples-per-agent",
        type=int,
        default=10,
        help="If >0, train a separate classifier per agent on this many examples",
    )
    args = parser.parse_args()

    run_simulation(
        steps=args.steps,
        n_agents=args.agents,
        m_links=args.m_links,
        influencer_ratio=args.influencers,
        bot_ratio=args.bots,
        fact_checker_ratio=args.fact_checkers,
        susceptible_ratio=args.susceptible,
        believer_ratio=args.believer,
        informed_ratio=args.informed,
        seed=args.seed,
        news_csv=args.news_csv,
        output_dir=args.output_dir,
        classifier_weight=args.classifier_weight,
        train_samples_per_agent=args.train_samples_per_agent,
        animate=not args.no_animate,
        show_graph=not args.no_graph,
        show_timeseries=args.timeseries,
        interval_ms=args.interval,
    )
