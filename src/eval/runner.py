# runs search agent on each question, judges and collects results.
import json
import random
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm
# keep track of API usage
from langchain_community.callbacks.manager import get_openai_callback

random.seed(42)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("data/results")
output_dir.mkdir(parents=True, exist_ok=True)


def run_eval(agent, judge, questions,  question_col="problem", gold_col="answer"):
    """Run agent on each question, judge the result and return list of records."""
    results = []
    for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Eval"):
        question = row[question_col]
        gold = row[gold_col]

        start = time.perf_counter()
        try:
            with get_openai_callback() as cb:
                result = agent(question)
            tool_uses = result["tool_uses"]
            predicted = result["answer"]
            agent_cost = cb.total_cost
            error = None
        except Exception as e:
            predicted = None
            tool_uses = []
            agent_cost = 0.0
            error = f"agent error : {e}"
        latency_s = time.perf_counter() - start

        # Run judge
        judge_cost = 0.0
        if predicted is not None:
            try:
                with get_openai_callback() as cb:
                    verdict = judge(question, gold, predicted)
                judge_cost = cb.total_cost
            except Exception as e:
                verdict = None
                error = f"Judge error : {e}"
        else:
            verdict = None
        results.append({
            "question": question,
            "gold": gold,
            "predicted": predicted,
            "tool_uses": tool_uses,
            "verdict": verdict,
            "latency_s": round(latency_s, 2),
            "agemt_cost": round(agent_cost, 5),
            "judge_cost": round(judge_cost, 5),
            "total_cost": round(agent_cost + judge_cost, 5),
            "error": error
        })
    return results


if __name__ == "__main__":
    from src.agents.with_search import ask
    from src.eval.judge import judge
    from src.eval.metrics import compute_metrics
    dev = pd.read_csv("data/simpleqa_verified.csv")
    dev_subset = dev.sample(n=100,  random_state=42)  # taking subset for now

    results = run_eval(agent=ask, judge=judge, questions=dev_subset)
    metrics = compute_metrics(results)
    metrics_path = output_dir / f"baseline_metrics_{timestamp}.json"
    result_path = output_dir / f"baseline_results_{timestamp}.json"

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Doneeeeeeeeeeeee")
