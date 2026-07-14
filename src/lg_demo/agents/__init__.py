from lg_demo.agents.arithmetic_agent import build_simple_arithmetic_agent
from lg_demo.agents.trading_agent import build_trading_agent
from lg_demo.agents.travel_agent import build_travel_agent


class _AgentRegistry:

    general_agent = build_simple_arithmetic_agent()
    finance_agent = build_trading_agent()
    travel_agent = build_travel_agent()


AgentRegistry = _AgentRegistry()
