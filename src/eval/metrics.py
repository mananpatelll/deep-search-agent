def compute_metrics(results):
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    not_attempted = sum(1 for r in results if r["verdict"] == "not_attempted")
    errors = sum(1 for r in results if r["verdict"] is None)

    accuracy = correct / total if total > 0 else 0.0
    total_cost = sum(r["cost_used"] for r in results)
    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "errors": errors,
        "accuracy": accuracy,
        "total_cost": total_cost
    }
