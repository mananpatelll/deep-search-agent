# LLM as judge, will help to evaluate agent performace
from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
model = ChatOpenAI(model="gpt-4o", temperature=0)


class Judgement(BaseModel):
    verdict: Literal["correct", "incorrect", "not_attempted"]


JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating answers to factual questions.

Compare the agent's response to the ground truth and return exactly one verdict:
"correct", "incorrect", or "not_attempted".
    "correct" - the agent gave exact answer as the ground truth. 
    "incorrect" - The agent gave an answer that does not match the ground truth.
    "not_attempted" - The agent explicitly refused or did not give any answer to the question, for example - "I don't know", "I cannot determine", "I don't have that information", etc..
"""
JUDGE_USER_PROMPT = """\
Question: {question}

Ground Truth: {ground_truth}

Agent Response: {response_to_judge}"""

_prompt_template = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM_PROMPT),
    ("user", JUDGE_USER_PROMPT),
])
_judge_model = model.with_structured_output(Judgement)


def judge(question: str, ground_truth: str, response_to_judge: str) -> Judgement:
    messages = _prompt_template.format_messages(
        question=question,
        ground_truth=ground_truth,
        response_to_judge=response_to_judge,
    )
    judgement = _judge_model.invoke(messages)
    return judgement.verdict
