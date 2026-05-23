# runs search agent on test dataset
import json
import random
from pathlib import Path
from datetime import datetime
import pandas as pd
random.seed(42)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("data/results")
output_dir.mkdir(parents=True, exist_ok=True)


def run_eval(agent, judge, questions,  question_col="problem", gold_col="answer"):
    results = []
    for _, row in questions.iterrows():
        question = row[question_col]
        gold = row[gold_col]

        predicted = agent(question)
        verdict = judge(question, gold, predicted)
        results.append({
            "question": question,
            "gold": gold,
            "predicted": predicted,
            "verdict": verdict,
        })
    return results


if __name__ == "__main__":
    from src.agent import ask
    from src.eval.judge import judge
    from src.eval.metrics import compute_metrics
    dev = pd.read_csv("data/simpleqa_verified.csv")
    dev_subset = dev.sample(n=20,  random_state=42)  # taking subset for now

    results = run_eval(agent=ask, judge=judge, questions=dev_subset)
    metrics = compute_metrics(results)
    metrics_path = output_dir / f"baseline_metrics_{timestamp}.json"
    result_path = output_dir / f"baseline_results_{timestamp}.json"

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Doneeeeeeeeeeeee")
