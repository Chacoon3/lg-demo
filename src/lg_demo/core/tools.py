import json
import os
from typing import Literal

from langchain.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


class ArithmeticOperation(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")


# Define tools
@tool(args_schema=ArithmeticOperation, description="Multiply two numbers")
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool(args_schema=ArithmeticOperation, description="Add two numbers")
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool(args_schema=ArithmeticOperation, description="Divide two numbers")
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


@tool(args_schema=ArithmeticOperation, description="Raise a number to the power of another number")
def power(a: int, b: float) -> float:
    """Raise `a` to the power of `b`.

    Args:
        a: Base integer
        b: Exponent
    """
    return a**b


class WebSearchResult(BaseModel):
    title: str
    url: str
    content: str


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResult]
    answer: str | None = None


_tavily = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)


@tool
def web_search(
    query: str,
    topic: Literal["general", "news", "finance"] = "general",
    time_range: Literal["day", "week", "month", "year"] | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> str:
    """Search the public web for current or external information.

    Use this tool when the answer depends on information that may have changed,
    when the user asks for recent news, or when factual claims require sources.

    Args:
        query: A precise natural-language web search query.
        max_results: Maximum number of search results to return, from 1 to 10.
        topic: Search category: general, news, or finance.
        time_range: Optional recency filter.
        include_domains: Optional domains to restrict results to.
        exclude_domains: Optional domains to exclude.

    Returns:
        A JSON string containing the search query, optional generated answer,
        and normalized search results with title, URL, and content.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return json.dumps(
            {
                "error": "TAVILY_API_KEY is not configured",
                "query": query,
            },
            ensure_ascii=False,
        )

    query = query.strip()
    if not query:
        return json.dumps(
            {"error": "query must not be empty"},
            ensure_ascii=False,
        )

    try:
        # Some parameters may be overridden at invocation time.
        response = _tavily.invoke(
            {
                "query": query,
                "topic": topic,
                "time_range": time_range,
                "include_domains": include_domains or [],
                "exclude_domains": exclude_domains or [],
            }
        )

        # Depending on package/version, invoke may return either a dict
        # or an already serialized JSON string.
        if isinstance(response, str):
            response = json.loads(response)

        if response.get("error"):
            raise RuntimeError(response.get("error"))

        normalized = WebSearchResponse(
            query=response.get("query", query),
            answer=response.get("answer"),
            results=[
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                )
                for item in response.get("results", [])
            ],
        )

        return normalized.model_dump_json()

    except Exception as exc:
        return json.dumps(
            {
                "error": f"web search failed: {type(exc).__name__}: {exc}",
                "query": query,
            },
            ensure_ascii=False,
        )
