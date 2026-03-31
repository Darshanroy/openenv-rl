"""Quick smoke test for OpenEnv compliance."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env.server.my_environment import SupportEnvironment
from my_env.models import SupportAction, SupportObservation, SupportState
from openenv.core import Environment

print("=" * 60)
print("  OpenEnv Compliance Smoke Test")
print("=" * 60)

# 1. Inheritance check
assert issubclass(SupportEnvironment, Environment), "FAIL: Not inheriting from openenv.core.Environment"
print("[PASS] SupportEnvironment inherits from openenv.core.Environment")

# 2. Reset
env = SupportEnvironment()
obs = env.reset(task_id="easy_status")
assert isinstance(obs, SupportObservation), "FAIL: reset() didn't return SupportObservation"
assert obs.done == False
assert obs.reward == 0.0
assert len(obs.messages) > 0
print(f"[PASS] reset() -> SupportObservation (done={obs.done}, msgs={len(obs.messages)})")
print(f"       Customer: {obs.messages[0]['content']}")

# 3. Step
action = SupportAction(message="[get_order('ORD-101')]")
obs2 = env.step(action)
assert isinstance(obs2, SupportObservation)
print(f"[PASS] step(get_order) -> reward={obs2.reward}, done={obs2.done}")
print(f"       Feedback: {obs2.messages[-1]['content'][:80]}...")

# 4. State property
state = env.state
assert isinstance(state, SupportState)
assert state.step_count == 1
assert "get_order" in state.tools_used
print(f"[PASS] state -> step_count={state.step_count}, tools={state.tools_used}")

# 5. Grader score at completion
obs3 = env.step(SupportAction(message="[track_shipment('ORD-101')]"))
obs4 = env.step(SupportAction(message="[respond('Your order is on its way!')]"))
assert obs4.done == True
grader = obs4.metadata.get("grader_score", -1)
assert 0.0 <= grader <= 1.0, f"FAIL: grader_score={grader} not in [0,1]"
print(f"[PASS] Episode complete: grader_score={grader:.2f}, steps={env.state.step_count}")

# 6. TRL SupportToolEnv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from support_tool_env import SupportToolEnv

tool_env = SupportToolEnv()
init_obs = tool_env.reset()
assert isinstance(init_obs, str), "FAIL: reset() should return a string"
print(f"\n[PASS] SupportToolEnv.reset() -> str ({len(init_obs)} chars)")

# Check tool methods are discoverable
import inspect
public_methods = [
    name for name, _ in inspect.getmembers(tool_env, predicate=inspect.ismethod)
    if not name.startswith("_") and name != "reset"
]
print(f"[PASS] Discovered {len(public_methods)} tool methods: {public_methods}")

# Run a tool
result = tool_env.get_order("ORD-101")
assert isinstance(result, str)
print(f"[PASS] get_order('ORD-101') -> {result[:60]}...")

# End episode
result = tool_env.respond("Your order is shipping!")
assert tool_env.done == True
assert 0.0 <= tool_env.reward <= 1.0
print(f"[PASS] respond() -> done={tool_env.done}, reward={tool_env.reward:.2f}")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED - OpenEnv Compliant!")
print("=" * 60)
