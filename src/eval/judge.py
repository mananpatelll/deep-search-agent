# LLM as judge, will help to evaluate agent performace
from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class Judgement(BaseModel):
    verdict: Literal["correct", "incorrect", "not_attempted"]


def judge(question: str, ground_truth: str, response_to_judge: str) -> Judgement:
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
                You are an impartial, highly accurate AI judge. Your task is to evaluate a given answer to a question.
                Compare the 'Response to Judge' against the 'Ground Truth' and 'Question'.
                Provide a verdict  from, 'correct' 'incorrect' or 'not_attempted'.
            """),
        (
            "user",
            """
                Question : {question}
                Ground Truth : {ground_truth}
                Response to Judge : {response_to_judge}
            """)
    ])
    formatted_messages = prompt_template.format_messages(
        question=question,
        ground_truth=ground_truth,
        response_to_judge=response_to_judge
    )
    response = model.with_structured_output(
        Judgement).invoke(formatted_messages)
    return response.verdict
