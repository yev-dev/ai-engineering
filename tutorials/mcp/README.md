# Model Context Protocol (MCP): Connecting AI to Tools and Data

The **Model Context Protocol (MCP)** is an open standard developed by Anthropic that defines how AI applications (hosts) connect to external tools, data sources, and services (servers). It solves a fundamental problem: LLMs are stateless and have no native access to real-time data, APIs, or file systems. MCP provides a unified, secure interface for bridging that gap.

## Why We Need MCP

Before MCP, every AI application integrated with external services using ad-hoc methods:

- **Custom API wrappers** — each tool required bespoke code for authentication, error handling, and response parsing.
- **Insecure prompt injection** — tool descriptions and credentials were often embedded in system prompts, creating security risks.
- **No standard discovery** — an AI app couldn't ask "what tools do you have?" without hardcoded knowledge.

MCP solves these problems by defining a **standard protocol** with three roles:

| Role | Description | Example |
|------|-------------|---------|
| **Host** | The AI application that needs tools (e.g. Claude Desktop, Cline, a custom app) | Cline, Claude Desktop |
| **Server** | A lightweight service that exposes tools, resources, and prompts | GitHub MCP server, filesystem server |
| **Client** | The bridge inside the host that connects to servers | MCP client library |

### Benefits of MCP

1. **Standardised tool interface** — Every server exposes tools via the same JSON-RPC protocol. The host doesn't care about the implementation language (Python, TypeScript, Go, etc.).
2. **Dynamic discovery** — The host can query `list_tools()` to see all available tools at runtime, with names, descriptions, and input schemas.
3. **Security isolation** — Servers run as separate processes with explicit permissions. A file-system server doesn't have network access unless configured.
4. **Reusable servers** — A single GitHub MCP server can be used by any MCP-compatible host. No need to rewrite the same integration for every app.
5. **Resource and prompt exposure** — Beyond tools, MCP servers can expose resources (files, API responses) and reusable prompt templates.

### MCP vs Other Approaches

| Approach | Standardisation | Security | Discovery | Language Agnostic |
|----------|----------------|----------|-----------|-------------------|
| Custom API wrapper | ❌ Per-app | ❌ Mixed | ❌ Hardcoded | ✅ Any language |
| LangChain tools | ⚠️ LangChain-only | ⚠️ Plugin-dependent | ⚠️ Registry-based | ❌ Python/JS only |
| **MCP** | **✅ Open standard** | **✅ Process isolation** | **✅ Dynamic** | **✅ Any language** |
| OpenAI function calling | ⚠️ OpenAI-only | ⚠️ API key based | ❌ Defined at call time | ❌ OpenAI ecosystem |

### How MCP Works

```
┌─────────────────┐       JSON-RPC over stdio/SSE       ┌──────────────────┐
│                 │◄───────────────────────────────────►│                  │
│   MCP Host      │   list_tools() → [tool definitions] │   MCP Server     │
│   (Cline,       │   call_tool(name, args) → result    │   (Python, TS)   │
│   Claude, etc)  │   list_resources() → [resources]    │                  │
│                 │                                      │  ┌────────────┐  │
│  ┌───────────┐  │                                      │  │ GitHub API │  │
│  │ LLM       │  │                                      │  │ Filesystem │  │
│  │ decides   │  │                                      │  │ Database   │  │
│  │ tool call │  │                                      │  │ ...        │  │
│  └───────────┘  │                                      │  └────────────┘  │
└─────────────────┘                                      └──────────────────┘
```

Communication happens over **JSON-RPC 2.0** through either:
- **stdio** — Server runs as a subprocess, communication via stdin/stdout (local, secure)
- **SSE (Server-Sent Events)** — Server runs as an HTTP endpoint (remote, can be deployed on a server)

## Example: GitHub MCP Server

Below is a minimal MCP server that exposes GitHub repository operations. The server is written in Python using the official `mcp` package.

### Server Code

```python
import os
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("GitHub Assistant")

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


@mcp.tool()
async def search_repositories(query: str, per_page: int = 5) -> str:
    """Search for GitHub repositories matching a query.
    
    Args:
        query: The search query (e.g. "quantum computing python")
        per_page: Number of results to return (max 30)
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GITHUB_API_BASE}/search/repositories",
            params={"q": query, "per_page": per_page, "sort": "stars"},
            headers=HEADERS,
        )
        response.raise_for_status()
        data = response.json()
    
    if not data["items"]:
        return "No repositories found."
    
    results = []
    for repo in data["items"]:
        results.append(
            f"• {repo['full_name']} — ⭐ {repo['stargazers_count']} — "
            f"{repo['description'] or 'No description'}"
        )
    return "\n".join(results)


@mcp.tool()
async def get_repository(owner: str, repo: str) -> str:
    """Get detailed information about a specific repository.
    
    Args:
        owner: Repository owner (user or organization)
        repo: Repository name
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}",
            headers=HEADERS,
        )
        response.raise_for_status()
        data = response.json()
    
    return (
        f"📦 {data['full_name']}\n"
        f"📝 {data['description'] or 'No description'}\n"
        f"⭐ Stars: {data['stargazers_count']}  🍴 Forks: {data['forks_count']}\n"
        f"🐛 Open Issues: {data['open_issues_count']}\n"
        f"📅 Created: {data['created_at'][:10]}  "
        f"🔄 Updated: {data['updated_at'][:10]}\n"
        f"🌐 {data['html_url']}"
    )


@mcp.tool()
async def list_issues(owner: str, repo: str, state: str = "open", per_page: int = 5) -> str:
    """List issues for a GitHub repository.
    
    Args:
        owner: Repository owner
        repo: Repository name
        state: Issue state (open, closed, all)
        per_page: Number of issues to return
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues",
            params={"state": state, "per_page": per_page, "sort": "updated"},
            headers=HEADERS,
        )
        response.raise_for_status()
        issues = response.json()
    
    if not issues:
        return f"No {state} issues found."
    
    results = []
    for issue in issues:
        if "pull_request" not in issue:  # Filter out PRs
            results.append(
                f"• #{issue['number']} {issue['title']} — "
                f"by {issue['user']['login']} ({issue['state']})"
            )
    return "\n".join(results) if results else f"No {state} issues found (excluding PRs)."


@mcp.tool()
async def get_readme(owner: str, repo: str) -> str:
    """Get the README content of a repository.
    
    Args:
        owner: Repository owner
        repo: Repository name
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme",
            headers={**HEADERS, "Accept": "application/vnd.github.raw+json"},
        )
        response.raise_for_status()
        return response.text[:2000]  # Truncate for context window


@mcp.resource("github://{owner}/{repo}/issues")
async def get_issues_resource(owner: str, repo: str) -> str:
    """Expose open issues as a readable resource."""
    return await list_issues(owner, repo, state="open", per_page=10)


if __name__ == "__main__":
    mcp.run()
```

### Running the Server

```bash
# Install the MCP package
pip install mcp httpx

# Set your GitHub token (optional, gives higher rate limits)
export GITHUB_TOKEN="ghp_youraccesstoken"

# Run the server over stdio (for local MCP hosts like Cline, Claude Desktop)
python github_mcp_server.py

# Or run as an HTTP SSE endpoint (for remote access)
python github_mcp_server.py --transport sse --port 8000
```

### Configuring with Cline

To use this server in Cline, add it to `.clinerules` or the MCP settings:

```json
{
  "mcpServers": {
    "github-assistant": {
      "command": "python",
      "args": ["/path/to/github_mcp_server.py"],
      "env": {
        "GITHUB_TOKEN": "ghp_youraccesstoken"
      }
    }
  }
}
```

## Deployment Options

MCP servers can be deployed in several ways depending on your needs:

### 1. Local stdio (Default)

The server runs as a subprocess of the MCP host. Communication is via stdin/stdout.

```
✅ Pros: Simple, no network overhead, secure (no open ports)
❌ Cons: Only accessible on the local machine, server must be installed locally
```

```bash
# The host spawns the server process
mcp run github_mcp_server.py
```

### 2. HTTP SSE Server

The server runs as a web server, accessible over HTTP. Multiple hosts can connect.

```
✅ Pros: Remote access, multiple clients, easy to scale
❌ Cons: Requires network configuration, security considerations (auth, HTTPS)
```

```bash
# Run as SSE server on port 8000
python github_mcp_server.py --transport sse --port 8000

# The host connects via URL
# In Cline config:
{
  "mcpServers": {
    "github-assistant": {
      "type": "url",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### 3. Docker Container

Package the server as a Docker image for consistent deployment.

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY github_mcp_server.py .

EXPOSE 8000
CMD ["python", "github_mcp_server.py", "--transport", "sse", "--port", "8000"]
```

```bash
docker build -t mcp-github .
docker run -e GITHUB_TOKEN=$GITHUB_TOKEN -p 8000:8000 mcp-github
```

### 4. Cloud Deployment (Fly.io, Railway, Render)

Deploy the SSE server to a cloud platform for persistent, publicly accessible endpoints.

```bash
# Example: deploy to Railway
railway login
railway init
railway up

# The server URL will be something like:
# https://mcp-github.up.railway.app/sse
```

### 5. GitHub Actions (CI/CD)

For automated workflows, the MCP server can run as part of a GitHub Action.

```yaml
name: MCP Server CI
on: [push]
jobs:
  test-mcp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install mcp httpx
      - run: python github_mcp_server.py &
      - run: |
          # Test the server with a simple MCP client
          echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
          python -c "
            import sys, json
            # Send a list_tools request
            sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':1,'method':'tools/list'}))
          "
```

## Creating Your Own MCP Server

The `mcp` Python package makes it easy to create servers:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool()
def my_tool(param1: str, param2: int = 10) -> str:
    """Tool description (becomes the LLM-facing description)."""
    return f"Result: {param1} * {param2}"

@mcp.resource("my://data/{item_id}")
def my_resource(item_id: str) -> str:
    """Resource description."""
    return f"Data for {item_id}"

if __name__ == "__main__":
    mcp.run()
```

### Key MCP Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Tool** | A function the LLM can call | `search_repositories(query)` |
| **Resource** | A data source exposed with a URI scheme | `github://owner/repo/issues` |
| **Prompt** | A reusable prompt template | A "code review" prompt template |
| **Transport** | How the server communicates | stdio (local) or SSE (remote) |

## Security Considerations

1. **Token management**: Never hardcode tokens. Use environment variables or secret managers.
2. **Least privilege**: Give the MCP server only the permissions it needs. A GitHub server doesn't need filesystem access.
3. **Input validation**: Sanitize all inputs before passing to external APIs.
4. **Rate limiting**: Implement rate limiting to prevent abuse of external APIs.
5. **HTTPS in production**: Always use HTTPS for remote SSE servers.

## Suggested Flow

1. Read the conceptual sections above.
2. Run the Jupyter notebook `mcp_server.ipynb` to see a live MCP server in action.
3. Modify the server to add your own tools.
4. Deploy the server using one of the deployment options above.
5. Connect it to your MCP host (Cline, Claude Desktop, etc.).