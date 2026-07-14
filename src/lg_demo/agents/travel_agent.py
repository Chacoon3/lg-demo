from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph

from agent_api.app_logging import get_logger
from lg_demo.core.nodes import InferenceNode, ToolNode
from lg_demo.core.router import DirectRouter, EntryRouter, ToolCallRouter
from lg_demo.core.runtime import RuntimeBuilder
from lg_demo.core.states import AgentPlan, RuntimeState
from lg_demo.core.tools import web_search
from lg_demo.utils.caching import AppDiskCache
from lg_demo.utils.dag import Dag


class TravelPlannerNode(InferenceNode):

    def __init__(self, name: str, model):
        model = model.with_structured_output(AgentPlan)
        super().__init__(name, model)

    @AppDiskCache.wrap
    def __call__(self, state: RuntimeState) -> RuntimeState:
        msg: AgentPlan = self.model.invoke([SystemMessage(content="""
Come up with a list of tasks that break down the user's request into actionable steps.
Give each task a concise and unique name and put the task instruction in the "description" field.
If a task depends on any other tasks, put the other tasks' names in the "dependencies" field.
""")] + state.messages)

        plan_json = msg.model_dump_json()

        dag = Dag(msg.tasks)
        get_logger().debug("agent_plan", plan_json=plan_json)
        return RuntimeState(
            messages=[AIMessage(content=plan_json)], llm_calls=1, tool_calls=0, state={"dag": dag}
        )


class TravelActionNode(InferenceNode):

    def __call__(self, state: RuntimeState) -> RuntimeState:
        sys_msg = SystemMessage(content="""
Your task is to execute the task given to you.
""")

        dag: Dag = state.state.get("dag", [])
        if not dag:
            raise ValueError("No task DAG found in the state.")

        msg_history = [sys_msg] + state.messages
        for task in dag:
            if task.status == "completed":
                continue
            msg_history.append(SystemMessage(content=f"Process the task: {task.description}"))
            resp = self.model.invoke(msg_history)
            msg_history.append(resp)
            task.status = "completed"
            return RuntimeState(messages=msg_history, llm_calls=1, tool_calls=0, state={"dag": dag})


def build_travel_agent(model: BaseChatModel) -> CompiledStateGraph:
    tools = ToolNode(name="travel_tools", tools=[web_search])  # Replace with actual tools as needed

    model = model.bind_tools(tools.tools)
    planner_node = TravelPlannerNode(name="planner_node", model=model)
    action_node = TravelActionNode(name="action_node", model=model)

    entry_route = EntryRouter(entry_node=planner_node)
    tool_call_router = ToolCallRouter(
        from_nodes=[planner_node, action_node],
        to_nodes=[tools],
    )
    action_route = DirectRouter(from_nodes=[planner_node], to_nodes=[action_node])

    agent = RuntimeBuilder(
        nodes=[planner_node, tools, action_node],
        routers=[entry_route, tool_call_router, action_route],
    ).build()

    return agent
