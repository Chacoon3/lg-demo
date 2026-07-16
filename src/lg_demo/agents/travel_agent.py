from concurrent.futures import ThreadPoolExecutor
from os import cpu_count

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph

from agent_api.app_logging import get_logger
from lg_demo.core.nodes import InferenceNode, ToolNode
from lg_demo.core.router import DirectRouter, EntryRouter, ToolCallRouter
from lg_demo.core.runtime import RuntimeBuilder
from lg_demo.core.states import AgentPlan, RuntimeState
from lg_demo.core.tools import web_search
from lg_demo.utils.dag import Dag


class TravelPlannerNode(InferenceNode):

    def __init__(self, name: str, model):
        model = model.with_structured_output(AgentPlan)
        super().__init__(name, model)

    def __call__(self, state: RuntimeState) -> RuntimeState:
        msg: AgentPlan = self.model.invoke([SystemMessage(content="""
Solve the user's problem by breaking it down into a directed acyclic graph (DAG) of tasks.
Each task should be a single, well-defined action that can be executed independently.
The tasks should be designed to achieve the user's goal in a logical sequence, with dependencies clearly defined.
The tasks should not require any external input beyond what is already available in the state.
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
Execute the task given to you.
""")

        dag: Dag = state.state.get("dag", [])
        if not dag:
            raise ValueError("No task DAG found in the state.")
        get_logger().info(
            "Executing tasks in DAG", task_count=len(dag), task_names=[task.name for task in dag]
        )

        msg_history = state.messages + [sys_msg]
        llm_call = 0
        task_output = {}
        for layer in dag.iter_layers():
            with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                futures = []
                for task in layer:
                    llm_call += 1
                    # parallel tasks can share the same message history up to this point, but each task gets its own copy of the history with its own task instruction appended.
                    msg_history_copy = list(msg_history)
                    msg_history_copy.append(
                        HumanMessage(content=f"Process the task {task.name}: {task.description}")
                    )
                    futures.append(executor.submit(self.model.invoke, msg_history_copy))
                for task, future in zip(layer, futures):
                    resp = future.result()
                    msg_history.append(resp)
                    task_output[task.name] = resp.content
                    task.status = "completed"

        task_output_string = "\n".join(
            f"{task_name}: {output}" for task_name, output in task_output.items()
        )
        summerize_msg = [SystemMessage(content=f"""
Summarize the task outputs and generate an answer to the user's request.
{task_output_string}
                """)]

        summary = self.model.invoke(summerize_msg)

        return RuntimeState(
            messages=msg_history + [summary], llm_calls=llm_call, tool_calls=0, state={"dag": dag}
        )


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
