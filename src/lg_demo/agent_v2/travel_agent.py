"""
agent loop:
user request -> summarize -> plan -> execute -> (tool) -> evaluate -> (re-plan or complete)
"""

from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from agent_api.app_logging import get_logger
from lg_demo.core.nodes import InferenceNode, ToolNode
from lg_demo.core.states import ChainOfAction, SnapshotState
from lg_demo.core.tools import web_search


class _SummaryNode(InferenceNode):

    def __call__(self, state: SnapshotState) -> SnapshotState:
        sys_prompt = """
Summarize the user's request into a concise and clear summary that captures the main goal and any relevant details.
"""

        msg_snapshot = list(state.messages)
        msg_snapshot.append(SystemMessage(content=sys_prompt))

        resp = self.model.invoke(msg_snapshot)

        new_state = SnapshotState(
            messages=state.messages + [resp],
            llm_calls=state.llm_calls + 1,
            tool_calls=state.tool_calls,
        )
        return new_state


class _PlanningNode(InferenceNode):

    def __init__(self, model, name: str = "PlanningNode"):
        super().__init__(name=name, model=model)
        self.model = model.with_structured_output(ChainOfAction)

    def __call__(self, state: SnapshotState) -> SnapshotState:
        sys_prompt = """
You are an AI agent that breaks down the user's request into a series of actionable tasks.
Each task should have a concise name that summarizes its purpose.
In the "description" field provide detailed instructions regarding how to execute the task so that a LLM can understand and perform it.
You should generate the tasks in order of execution.
Provide the output as a JSON array of task objects.
"""

        resp: ChainOfAction = self.model.invoke(
            state.messages + [SystemMessage(content=sys_prompt)]
        )
        return SnapshotState(
            messages=state.messages + [resp],
            llm_calls=state.llm_calls + 1,
            tool_calls=state.tool_calls,
            state={"plan": resp.steps},
        )


class _ActionNode(InferenceNode):

    def __call__(self, state: SnapshotState) -> SnapshotState:
        sys_prompt = """
You are an agent that executes tasks.
Each task has a name and a description.
The description contains detailed instructions on how to execute the task.
"""

        chain_of_actions_data = state.state.get("plan", [])
        if not chain_of_actions_data:
            raise ValueError("No plan found in the state.")
        chain_of_actions = ChainOfAction(steps=chain_of_actions_data)
        action_index = chain_of_actions.next_step
        action = chain_of_actions.steps[action_index]
        msg = state.messages + [
            SystemMessage(content=sys_prompt),
            SystemMessage(content=action.description),
        ]
        # TODO try catch here
        resp = self.model.invoke(msg)
        chain_of_actions.set_current_step_state("completed")
        return SnapshotState(
            messages=state.messages + [resp],
            llm_calls=state.llm_calls + 1,
            tool_calls=state.tool_calls,
            state=chain_of_actions,
        )


class _EvaluationNode(InferenceNode):

    def __init__(self, model, name: str = "PlanningNode"):
        super().__init__(name=name, model=model)
        self.model = model.with_structured_output(ChainOfAction)

    def __call__(self, state: SnapshotState) -> SnapshotState:
        sys_prompt = """
Your goal is to analyze the current execution state of the list of tasks and determine if any modification should be made to the original plan.
Note that you can only modify the tasks which are failed or pending.
You cannot modify the tasks which are completed.
For the failed or pending tasks, you can modify their description, change their order, or insert new tasks to the plan.
You should provide the output as a JSON array of task objects.
In case no modification is needed, return the original plan without any changes.
"""

        # TODO consider summerizing the previous history here

        msg_snapshot = list(state.messages)
        msg_snapshot.append(SystemMessage(content=sys_prompt))

        resp: ChainOfAction = self.model.invoke(msg_snapshot)
        return SnapshotState(
            messages=state.messages + [resp],
            llm_calls=state.llm_calls + 1,
            tool_calls=state.tool_calls,
            state=resp,
        )


def should_continue_action(state: SnapshotState) -> str:
    cot: ChainOfAction = state.state
    if all(step.status == "completed" for step in cot.steps):
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


def build_cot_agent(model: BaseChatModel):
    summary_node = _SummaryNode(name="SummaryNode", model=model)
    planning_node = _PlanningNode(name="PlanningNode", model=model)
    action_node = _ActionNode(name="ActionNode", model=model)
    evaluation_node = _EvaluationNode(name="EvaluationNode", model=model)
    tool_node = ToolNode(name="ToolNode", tools=[web_search])  # Replace with actual tools as needed

    graph = StateGraph(SnapshotState)
    graph.add_node("should_continue_action", should_continue_action)
    nodes = [summary_node, planning_node, action_node, evaluation_node, tool_node]
    for n in nodes:
        graph.add_node(n.name, n)

    graph.add_edge(START, summary_node.name)
    graph.add_edge(summary_node.name, planning_node.name)
    graph.add_edge(planning_node.name, action_node.name)
    graph.add_conditional_edges(
        evaluation_node.name, should_continue_action, ["should_continue_action", END]
    )
    graph.add_conditional_edges(
        action_node.name,
        action_should_call_tool,
        ["ToolNode", "EvaluationNode"],
    )

    return graph.compile()
