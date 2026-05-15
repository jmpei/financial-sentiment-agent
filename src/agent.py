"""
Financial sentiment agent.

Pipeline: user question -> search_news -> analyze_sentiment per article ->
synthesized answer with sentiment distribution.

Public entry point: `run(question: str) -> str`
Bottom of file: REPL for interactive use.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.prompts import SYSTEM_PROMPT
from src.tools import analyze_sentiment, search_news

load_dotenv()


def _build_executor(verbose: bool = True) -> AgentExecutor:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set - fill .env first.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",      SYSTEM_PROMPT),
        ("human",       "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [search_news, analyze_sentiment]
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=15,
    )


_executor: AgentExecutor | None = None


def run(question: str) -> str:
    """Answer one financial question. Builds the executor on first call (lazy)."""
    global _executor
    if _executor is None:
        _executor = _build_executor(verbose=True)
    result = _executor.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    print("Financial Sentiment Agent. Type 'quit' or 'exit' to leave.")
    while True:
        try:
            q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if q.lower() in {"quit", "exit"}:
            sys.exit(0)
        if not q:
            continue
        try:
            print("\n" + run(q))
        except Exception as e:
            print(f"[error] {e}")
