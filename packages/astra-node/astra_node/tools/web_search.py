"""WebSearchTool — performs a web search via DuckDuckGo's Instant Answer API.

Permission: ASK_USER (makes an external network request).

Uses DuckDuckGo's free, no-key-required JSON API for basic search results.
This is suitable for development and lightweight use. Production deployments
should swap in a proper search API (Brave, Tavily, etc.) by subclassing.
"""

from pydantic import BaseModel, Field

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult


class WebSearchInput(BaseModel):
    """Input schema for WebSearchTool."""

    query: str = Field(..., description="The search query string.")
    num_results: int = Field(
        default=5,
        description="Number of results to return. Default 5.",
    )


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo and return result titles and URLs.

    Uses DuckDuckGo's free Instant Answer API. Returns a formatted list of
    results with title, URL, and snippet. No API key required.
    """

    name = "web_search"
    description = (
        "Search the web for information and return a list of relevant results. "
        "Each result includes a title, URL, and brief description."
    )
    input_schema = WebSearchInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: WebSearchInput, ctx: ToolContext) -> ToolResult:
        """Search DuckDuckGo for the query.

        Args:
            input: Validated WebSearchInput with query and num_results.
            ctx: Tool context (unused for HTTP requests).

        Returns:
            ToolResult with formatted search results, or an error on failure.
        """
        import json
        import urllib.parse
        import urllib.request
        import urllib.error

        try:
            encoded_query = urllib.parse.quote_plus(input.query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1&no_html=1"

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "astra-node/0.1 (agent framework)"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))

            results: list[str] = []

            # Abstract (featured answer)
            if data.get("AbstractText") and data.get("AbstractURL"):
                results.append(
                    f"[Featured] {data['AbstractText'][:300]}\n  URL: {data['AbstractURL']}"
                )

            # Related topics
            for topic in data.get("RelatedTopics", []):
                if len(results) >= input.num_results:
                    break
                if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                    results.append(
                        f"{topic['Text'][:200]}\n  URL: {topic['FirstURL']}"
                    )

            if not results:
                return ToolResult.ok(
                    f"No results found for query: '{input.query}'. "
                    "Try a different query or use web_fetch to fetch a specific URL."
                )

            output = f"Search results for: '{input.query}'\n\n"
            output += "\n\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
            return ToolResult.ok(output)

        except urllib.error.URLError as exc:
            return ToolResult.err(f"Search request failed: {exc.reason}")
        except Exception as exc:
            return ToolResult.err(f"Search failed: {exc}")
