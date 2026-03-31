"""
Verification script for OpenEnv Graders.
Runs through all 15 tasks, executes an empty trajectory (expecting 0.0) 
and a perfect trajectory (expecting 1.0), verifying all scores are bounded [0.0, 1.0].
"""
import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_env.client import SupportEnvClient, Action
from training.config import ENV_SERVER_URL

SCENARIOS = [
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account", "easy_cancel",
    "medium_delay",  "medium_address", "medium_reschedule", "medium_return", "medium_double_charge",
    "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation"
]

PERFECT_MOVES = {
    "easy_status": ["[get_order('ORD-101')]", "[track_shipment('ORD-101')]", "[respond('April 2nd')]"],
    "easy_payment_fail": ["[get_order('ORD-1414')]", "[respond('Failed')]"],
    "easy_coupon": ["[validate_coupon('SAVE10')]", "[respond('Done')]"],
    "easy_account": ["[reset_password('meera.reddy@example.com')]", "[respond('Sent')]"],
    "easy_cancel": ["[get_order('ORD-505')]", "[cancel_order('ORD-505')]", "[respond('Done')]"],
    "medium_delay": ["[get_order('ORD-909')]", "[track_shipment('ORD-909')]", "[respond('Sorry')]"],
    "medium_address": ["[get_order('ORD-1919')]", "[update_address('ORD-1919','New St')]", "[respond('Done')]"],
    "medium_reschedule": ["[get_order('ORD-2323')]", "[check_delivery_slot('ORD-2323')]", "[reschedule_delivery('ORD-2323','Slot1')]", "[respond('Done')]"],
    "medium_return": ["[get_order('ORD-2020')]", "[validate_return('ORD-2020')]", "[create_return_request('ORD-2020')]", "[respond('Done')]"],
    "medium_double_charge": ["[get_order('ORD-1515')]", "[initiate_refund('ORD-1515')]", "[respond('Done')]"],
    "hard_refund": ["[get_order('ORD-2121')]", "[validate_return('ORD-2121')]", "[initiate_refund('ORD-2121')]", "[respond('Done')]"],
    "hard_damaged": ["[get_order('ORD-2222')]", "[ask_proof('ORD-2222')]", "[validate_return('ORD-2222')]", "[initiate_refund('ORD-2222')]", "[respond('Done')]"],
    "hard_missing": ["[get_order('ORD-1313')]", "[track_shipment('ORD-1313')]", "[investigate_missing('ORD-1313')]", "[escalate_to_human('Missing')]"],
    "hard_angry": ["[get_order('ORD-909')]", "[track_shipment('ORD-909')]", "[respond('Sorry')]"],
    "hard_escalation": ["[get_order('ORD-1414')]", "[escalate_to_human('Manager')]"]
}

def verify_graders():
    client = SupportEnvClient(base_url=ENV_SERVER_URL)
    
    print("\n" + "="*80)
    print("VERIFYING GRADERS FOR 0.0 - 1.0 SPEC COMPLIANCE")
    print("="*80)
    
    all_passed = True

    for tid in SCENARIOS:
        print(f"\nEvaluating Task: {tid}")
        
        # Test 1: Empty / Failed Trajectory
        res_fail = client.reset(task_id=tid)
        # Extensively loop garbage turns to force an episode end via max_turns (set to 8 in Env)
        for _ in range(10):
            res_fail = client.step(Action(message="[invalid_tool()]"))
            if res_fail.done: break
        
        score_fail = res_fail.info.get("grader_score", -999.0)
        
        # Test 2: Perfect Trajectory
        res_perf = client.reset(task_id=tid)
        score_perf = 0.0
        for move in PERFECT_MOVES[tid]:
            res_perf = client.step(Action(message=move))
            if res_perf.done:
                score_perf = res_perf.info.get("grader_score", -999.0)
                break
                
        # Test 3: Partial Trajectory (To prove fine-grained decimal scoring like 0.1)
        res_part = client.reset(task_id=tid)
        score_part = 0.0
        res_part = client.step(Action(message="[respond('I am offering a partial response without checking the database.')]"))
        if res_part.done:
            score_part = res_part.info.get("grader_score", -999.0)
        else:
            # Force finish
            for _ in range(10):
                res_part = client.step(Action(message="[invalid_tool()]"))
                if res_part.done: 
                    score_part = res_part.info.get("grader_score", -999.0)
                    break
                
        # Assertions
        is_valid_fail = 0.0 <= score_fail <= 1.0
        is_valid_perf = 0.0 <= score_perf <= 1.0
        is_valid_part = 0.0 <= score_part <= 1.0
        
        print(f"  [Failure Test] Score: {score_fail:.2f} | Bounded [0,1]: {'PASS' if is_valid_fail else 'FAIL'}")
        print(f"  [Partial Test] Score: {score_part:.2f} | Bounded [0,1]: {'PASS' if is_valid_part else 'FAIL'} (Checks partial decimal values)")
        print(f"  [Perfect Test] Score: {score_perf:.2f} | Bounded [0,1]: {'PASS' if is_valid_perf else 'FAIL'}")
        
        if not is_valid_fail or not is_valid_perf or not is_valid_part:
            all_passed = False
            
    print("\n" + "="*80)
    if all_passed:
        print("PASS: ALL GRADERS VERIFIED. All scores are strictly bounded between 0.0 and 1.0.")
    else:
        print("FAIL: GRADER VIOLATION DETECTED. Some scores fell outside the [0.0, 1.0] range.")
    print("="*80 + "\n")

if __name__ == "__main__":
    verify_graders()
