from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.prompts import CORE_PROMPT, RAW_LLM_ADDENDUM
load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPT = CORE_PROMPT + RAW_LLM_ADDENDUM


def ask(question: str) -> dict:
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system", SYSTEM_PROMPT
        ),
        (
            "user",
            """
              {question}
            """
        )
    ])
    formatted_messages = prompt_template.format_messages(
        question=question
    )
    response = model.invoke(formatted_messages)
    final_ans = response.content
    return {
        "answer": final_ans,
        "tool_uses": []
    }


if __name__ == "__main__":
    while True:
        question = input("Question (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit", "q", ""}:
            break
        print(ask(question))
        print()
