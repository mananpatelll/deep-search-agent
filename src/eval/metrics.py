def compute_metrics(results):
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    not_attempted = sum(1 for r in results if r["verdict"] == "not_attempted")
    errors = sum(1 for r in results if r["verdict"] is None)
    attempted = correct + incorrect
    attempted_accuracy = correct / attempted if attempted > 0 else 0.0
    attempt_rate = attempted / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    total_agent_cost = sum(r["agent_cost"] for r in results)
    total_judge_cost = sum(r["judge_cost"] for r in results)
    total_cost = sum(r["total_cost"] for r in results)
    latencies = [r["latency_s"]
                 for r in results if r.get("latency_s") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = sorted(latencies)[int(
        len(latencies) * 0.95)] if latencies else 0.0
    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "errors": errors,
        "accuracy": accuracy,
        "attempted": attempted,
        "attempted_accuracy": attempted_accuracy,
        "attempt_rate": attempt_rate,
        "total_agent_cost": total_agent_cost,
        "total_judge_cost": total_judge_cost,
        "total_cost": total_cost,
        "avg_latency_s": round(avg_latency, 2),
        "p95_latency_s": round(p95_latency, 2)

    }
