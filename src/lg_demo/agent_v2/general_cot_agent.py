"""
agent loop:
user request -> summarize -> plan -> execute -> (tool) -> evaluate -> (re-plan or complete)
"""

from functools import partial

from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agent_api.app_logging import get_logger
from lg_demo.core.states import ChainOfAction, SnapshotState
from lg_demo.core.tools import web_search


def _summary_node(state: SnapshotState, model: BaseChatModel) -> SnapshotState:
    sys_prompt = """
Summarize the user's request into a concise and clear summary that captures the main goal and any relevant details.
"""

    msg_snapshot = list(state.messages)
    msg_snapshot.append(SystemMessage(content=sys_prompt))

    resp = model.invoke(msg_snapshot)

    return SnapshotState(
        messages=state.messages + [resp],
        llm_calls=state.llm_calls + 1,
        tool_calls=state.tool_calls,
    )


def _planning_node(state: SnapshotState, model: BaseChatModel) -> SnapshotState:
    sys_prompt = """
You are an AI agent that breaks down the user's request into a series of actionable tasks.
Each task should have a concise name that summarizes its purpose.
In the "description" field provide detailed instructions regarding how to execute the task so that a LLM can understand and perform it.
You should generate the tasks in order of execution.
Provide the output as a JSON array of task objects.
"""

    structured_model = model.with_structured_output(ChainOfAction)
    cot: ChainOfAction = structured_model.invoke(
        state.messages + [SystemMessage(content=sys_prompt)]
    )
    return SnapshotState(
        messages=state.messages,
        llm_calls=state.llm_calls + 1,
        tool_calls=state.tool_calls,
        state=cot,
    )


def _action_node(state: SnapshotState, model: BaseChatModel) -> SnapshotState:
    sys_prompt = """
You are an agent that executes tasks.
Each task has a name and a description.
The description contains detailed instructions on how to execute the task.
"""

    chain_of_actions: ChainOfAction = state.state
    if not chain_of_actions:
        raise ValueError("No plan found in the state.")
    action_index = chain_of_actions.next_step
    action = chain_of_actions.steps[action_index]
    msg = state.messages + [
        SystemMessage(content=sys_prompt),
        SystemMessage(content=action.description),
    ]
    resp = model.invoke(msg)
    chain_of_actions.set_current_step_state("completed")
    return SnapshotState(
        messages=state.messages + [resp],
        llm_calls=state.llm_calls + 1,
        tool_calls=state.tool_calls,
        state=chain_of_actions,
    )


def _evaluation_node(state: SnapshotState, model: BaseChatModel) -> SnapshotState:
    sys_prompt = """
Your goal is to analyze the current execution state of the list of tasks and determine if any modification should be made to the original plan.
Note that you can only modify the tasks which are failed or pending.
You cannot modify the tasks which are completed.
For the failed or pending tasks, you can modify their description, change their order, or insert new tasks to the plan.
You should provide the output as a JSON array of task objects.
In case no modification is needed, return the original plan without any changes.
"""

    msg_snapshot = list(state.messages)
    msg_snapshot.append(SystemMessage(content=sys_prompt))

    structured_model = model.with_structured_output(ChainOfAction)
    resp: ChainOfAction = structured_model.invoke(msg_snapshot)
    return SnapshotState(
        messages=state.messages,
        llm_calls=state.llm_calls + 1,
        tool_calls=state.tool_calls,
        state=resp,
    )


def should_continue_action(state: SnapshotState) -> str:
    cot: ChainOfAction = state.state
    if all(step.state == "completed" for step in cot.steps):
        return END

    if state.llm_calls >= 20:
        get_logger().warning("Maximum LLM calls reached, stopping execution.")
        return END

    return "ActionNode"


def action_should_call_tool(state: SnapshotState) -> str:
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "ToolNode"
    return "EvaluationNode"


def _tool_node(state: SnapshotState) -> SnapshotState:
    result = []
    for tool_call in state.messages[-1].tool_calls:
        if tool_call["name"] != web_search.name:
            raise ValueError(f"Unsupported tool: {tool_call['name']}")
        observation = web_search.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return SnapshotState(
        messages=result,
        llm_calls=0,
        tool_calls=len(result),
        state=state.state,
    )


def build_cot_agent(model: BaseChatModel):
    summary_node = partial(_summary_node, model=model)
    planning_node = partial(_planning_node, model=model)
    action_node = partial(_action_node, model=model)
    evaluation_node = partial(_evaluation_node, model=model)

    graph = StateGraph(SnapshotState)
    graph.add_node("should_continue_action", should_continue_action)
    graph.add_node("SummaryNode", summary_node)
    graph.add_node("PlanningNode", planning_node)
    graph.add_node("ActionNode", action_node)
    graph.add_node("EvaluationNode", evaluation_node)
    graph.add_node("ToolNode", _tool_node)

    graph.add_edge(START, "SummaryNode")
    graph.add_edge("SummaryNode", "PlanningNode")
    graph.add_edge("PlanningNode", "ActionNode")
    graph.add_conditional_edges("EvaluationNode", should_continue_action, ["ActionNode", END])
    graph.add_conditional_edges(
        "ActionNode",
        action_should_call_tool,
        ["ToolNode", "EvaluationNode"],
    )

    return graph.compile()
