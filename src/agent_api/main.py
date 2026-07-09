"""FastAPI application entry point."""

from fastapi.concurrency import asynccontextmanager
from langchain.messages import SystemMessage
from langchain_ollama import ChatOllama
import uvicorn
from fastapi import FastAPI
from agent_api.api import agent_api_router
from lg_demo.tools import divide, add, multiply

app = FastAPI(title="LG Demo Agent API", version="0.1.0")


@asynccontextmanager
async def lifespan(app: FastAPI):

    model = ChatOllama(
        model="qwen3.6",
        temperature=0,
        num_gpu=1,
    )

    # Augment the LLM with tools
    tools = [add, multiply, divide]
    model_with_tools = model.bind_tools(tools)

    def llm_call(state: dict):
        """LLM decides whether to call a tool or not"""

        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    pass


app.include_router(
    prefix="/agent_api",
    tags=["agent_api"],
    router=agent_api_router,
)

if __name__ == "__main__":

    uvicorn.run("agent_api.main:app", host="127.0.0.1", port=8000, reload=True)
