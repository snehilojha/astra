"""WebFetchTool — fetches the content of a URL.

Permission: ASK_USER (makes an external network request).
"""

import ipaddress
import socket
import urllib.parse

from pydantic import BaseModel, Field

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult

_MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB hard cap


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Return (True, '') if the URL is safe to fetch, else (False, reason)."""
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False, f"Scheme '{parsed.scheme}' is not allowed; only http and https are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL has no hostname."

    # Resolve hostname to IP and block private/loopback ranges (SSRF prevention).
    try:
        ip_str = socket.getaddrinfo(hostname, None)[0][4][0]
        ip = ipaddress.ip_address(ip_str)
    except (socket.gaierror, ValueError):
        return False, f"Could not resolve hostname: {hostname}"

    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
        return False, f"Requests to private/internal addresses are not allowed ({ip})."

    return True, ""


class WebFetchInput(BaseModel):
    """Input schema for WebFetchTool."""

    url: str = Field(..., description="The URL to fetch.")
    max_length: int = Field(
        default=20000,
        description="Maximum number of characters to return. Default 20000.",
    )


class WebFetchTool(BaseTool):
    """Fetch the content of a URL and return it as text.

    Returns the raw response body, truncated to max_length characters.
    For HTML pages, returns the raw HTML (not rendered text).
    """

    name = "web_fetch"
    description = (
        "Fetch the content of a URL and return it as text. "
        "Useful for reading documentation, APIs, or web pages. "
        "Returns raw content truncated to max_length characters."
    )
    input_schema = WebFetchInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: WebFetchInput, ctx: ToolContext) -> ToolResult:
        """Fetch the URL using urllib.

        Args:
            input: Validated WebFetchInput with url and max_length.
            ctx: Tool context (unused for HTTP requests).

        Returns:
            ToolResult with the response body text, or an error on failure.
        """
        import urllib.request
        import urllib.error

        safe, reason = _is_safe_url(input.url)
        if not safe:
            return ToolResult.err(f"Blocked URL: {reason}")

        try:
            req = urllib.request.Request(
                input.url,
                headers={"User-Agent": "astra-node/0.1 (agent framework)"},
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                raw = response.read(_MAX_RESPONSE_BYTES)
                text = raw.decode("utf-8", errors="replace")
                if len(text) > input.max_length:
                    text = (
                        text[: input.max_length]
                        + f"\n... (truncated at {input.max_length} chars)"
                    )
                return ToolResult.ok(text)
        except urllib.error.HTTPError as exc:
            return ToolResult.err(f"HTTP {exc.code}: {exc.reason} — {input.url}")
        except urllib.error.URLError as exc:
            return ToolResult.err(f"URL error: {exc.reason}")
        except Exception as exc:
            return ToolResult.err(f"Failed to fetch {input.url}: {exc}")
