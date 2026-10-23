from agentic_hematology.agent_controller import AgentDecision, AgentAction

class FakeReflectionAgent:
    def __init__(self):
        self.calls = 0

    def decide(
        self,
        case_state,
        re_aggregate_used,
        current_conf_threshold,
        iteration,
    ):
        self.calls += 1

        print(f"Reflection call {self.calls}")

        if self.calls == 1:
            return AgentDecision(
                AgentAction.RE_AGGREGATE,
                "Testing re-aggregation",
                conf_threshold=0.5,
            )

        return AgentDecision(
            AgentAction.PROCEED,
            "Testing proceed",
        )