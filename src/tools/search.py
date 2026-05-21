import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()

search_tool = DuckDuckGoSearchResults(
    num_results=5,
    output_format="list"
)

agent = create_agent(
    model=ChatOpenAI(),
    tools=[search_tool],
    system_prompt=(
        "You are a research assistant. When a user asks a question,"
        "use the search tool to find current information, then sythesize"
        "a clear, factual answer. Always cite the sources you used"
    ),
)


def ask(question: str) -> str:
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    while True:
        question = input("Question : ")
        print(ask(question))
