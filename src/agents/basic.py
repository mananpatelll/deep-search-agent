# Basic agent without any custom framework for eval

"""Basic single-tool research agent."""
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()

search_tool = DuckDuckGoSearchResults(
    num_results=5,
    output_format="list",
)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    tools=[search_tool],
    system_prompt=(
        "You are a research assistant. When a user asks a question, "
        "use the search tool to find current information, then synthesize "
        "a clear, factual answer. Always cite the sources you used."
    ),
)


def ask(question: str) -> str:
    """Run the agent on a single question, return the final answer."""
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    final_ans = result["messages"][-1].content
    tool_uses = []  # Extract tool calls and their results from the message histort
    for msg in result["messages"]:
        # Tool call
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_uses.append(
                    {"type": "call", "tool": tc["name"], "args": tc["args"]})
            # Tool result
            if msg.__class__.__name__ == "ToolMessage":
                tool_uses.append({"type": "result", "conent": msg.content})
    return {
        "answer": final_ans,
        "tool_uses": tool_uses
    }


if __name__ == "__main__":
    while True:
        question = input("Question (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit", "q", ""}:
            break
        print(ask(question))
        print()
