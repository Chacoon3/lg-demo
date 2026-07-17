from langchain.chat_models import BaseChatModel

from lg_demo.agent_v2.travel_agent import build_cot_agent
from lg_demo.agents.arithmetic_agent import build_simple_arithmetic_agent
from lg_demo.agents.trading_agent import build_trading_agent
from lg_demo.agents.travel_agent import build_travel_agent


class AgentRegistry:

    def __init__(self, model: BaseChatModel):
        self.general_agent = build_simple_arithmetic_agent(model)
        self.finance_agent = build_trading_agent(model)
        self.travel_agent = build_travel_agent(model)
        self.general_cot = build_cot_agent(model)
