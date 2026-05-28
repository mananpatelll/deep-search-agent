"""CLI for running agent evaluations."""
import argparse
import importlib
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.eval.runner import run_eval
from src.eval.judge import judge
from src.eval.metrics import compute_metrics


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("data/results")
output_dir.mkdir(parents=True, exist_ok=True)

AGENT_REGISTRY = {
    "raw_llm": "src.agents.raw_llm",
    "with_search": "src.agents.with_search"
}


def load_agent(name: str):
    """import the named agent's ask() function"""
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent : {name}. Available : {list(AGENT_REGISTRY.keys())}"
        )
    module = importlib.import_module(AGENT_REGISTRY[name])
    return module.ask


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument(
        "--agent",
        required=True,
        choices=list(AGENT_REGISTRY.keys()),
        help="Which agent to evaluate on"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of questions to sample (default : 5)"
    )

    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Directory to write results into (default : data/results)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load agent
    print(f"Loading agent : {args.agent}")
    ask = load_agent(args.agent)

    # Load and sampel the data
    print(f"Loading data")
    dev = pd.read_csv("data/simpleqa_verified.csv")

    # Sample
    if args.n >= len(dev):
        questions = dev
        print(f"Using full dataset :{len(questions)} question")
    else:
        questions = dev.sample(n=args.n, random_state=42)
        print(f"Sampled {len(questions)} questions with seed 42")

    # Set up output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = output_dir / \
        f"{args.agent}_metrics_n{args.n}_{timestamp}.json"
    result_path = output_dir / \
        f"{args.agent}_results_n{args.n}_{timestamp}.json"

    # Run eval
    print(f"Runnig eval on {args.agent}")

    results = run_eval(agent=ask, judge=judge, questions=questions)

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["agent"] = args.agent
    metrics["n_sampled"] = args.n
    metrics["timestamp"] = timestamp

    # Save
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Doneeeeeeeeeeeee")


if __name__ == "__main__":
    main()
