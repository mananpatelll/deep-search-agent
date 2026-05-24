def compute_metrics(results):
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    not_attempted = sum(1 for r in results if r["verdict"] == "not_attempted")
    errors = sum(1 for r in results if r["verdict"] is None)

    accuracy = correct / total if total > 0 else 0.0
    total_agent_cost = sum(r["agemt_cost"] for r in results)
    total_judge_cost = sum(r["judge_cost"] for r in results)
    total_cost = sum(r["total_cost"] for r in results)
    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "errors": errors,
        "accuracy": accuracy,
        "total_agent_cost": total_agent_cost,
        "total_judge_cost": total_judge_cost,
        "total_cost": total_cost

    }
