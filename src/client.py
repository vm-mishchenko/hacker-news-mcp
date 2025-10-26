import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    # Connect to the server using Streamable HTTP
    async with streamablehttp_client("http://localhost:8000/mcp") as (
            read_stream,
            write_stream,
            get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Call Hacker News search tool
            result = await session.call_tool("HackerNewsSearchTool",
                                             arguments={"query": "mongodb"})
            print(f"Hacker News Search Results:\n{result.content}")


if __name__ == "__main__":
    asyncio.run(main())
