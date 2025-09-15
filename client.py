from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio
import os

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"], ## Ensure correct absolute path
                "transport":"stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",  # Ensure server is running here
                "transport": "streamable_http",
            },
            # Slack MCP server using the official package. Provide only the envs you have.
            # Required envs: SLACK_BOT_TOKEN, SLACK_TEAM_ID, SLACK_CHANNEL_IDS
            "slack": {
                "command": ("npx"),
                "args": ["-y", os.getenv("SLACK_MCP_PACKAGE", "@modelcontextprotocol/server-slack")],
                "transport": "stdio",
                "env": {
                    "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN", ""),
                    "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID", ""),
                    "SLACK_CHANNEL_IDS": os.getenv("SLACK_CHANNEL_IDS", "")
                }
            }
        }
    )

    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    tools=await client.get_tools()
    model=ChatGroq(model="openai/gpt-oss-120b")
    agent=create_react_agent(
        model,tools
    )

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )

    print("Math response:", math_response['messages'][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in San Francisco?"}]}
    )
    print("Weather response:", weather_response['messages'][-1].content)

    # Example Slack usage prompt. The agent will pick the right Slack tool if available.
    # Adjust the prompt or tool call based on your Slack MCP's exposed tools.
    slack_test_prompt = "Post the message 'Hello from MCP server' in the #cloud_softude_test channel."
    slack_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": slack_test_prompt}]}
    )
    print("Slack response:", slack_response['messages'][-1].content)

asyncio.run(main())