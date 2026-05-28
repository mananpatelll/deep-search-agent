# runs search agent on each question, judges and collects results.
import random
import time
from tqdm import tqdm
# keep track of API usage
from langchain_community.callbacks.manager import get_openai_callback

random.seed(42)


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
            "agent_cost": round(agent_cost, 5),
            "judge_cost": round(judge_cost, 5),
            "total_cost": round(agent_cost + judge_cost, 5),
            "error": error
        })
    return results
