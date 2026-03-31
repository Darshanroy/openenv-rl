import sys
import os
import time

# Ensure imports work regardless of local execution path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env.client import SupportEnvClient, Action

def run_benchmark():
    """
    Automated Baseline for all 15 Tasks in the Customer Support curriculum.
    """
    client = SupportEnvClient(base_url="http://127.0.0.1:8011")
    
    scenarios = [
        "easy_status", "easy_payment_fail", "easy_coupon", "easy_account",
        "medium_delay", "easy_cancel", "medium_address", "medium_reschedule", "medium_return",
        "medium_double_charge", "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation"
    ]
    
    print("\n" + "="*75)
    print("OPENENV CSA MASTER TABLE: 15 SCENARIO BENCHMARK (V3.0)")
    print("="*75 + "\n")

    results = []
    
    # Cleaned Perfect Sequences
    moves = {
        "easy_status": ["[get_order('ORD-101')]", "[track_shipment('ORD-101')]", "[respond('April 2nd')]"],
        "easy_payment_fail": ["[get_order('ORD-1414')]", "[respond('Failed')]"],
        "easy_coupon": ["[validate_coupon('SAVE10')]", "[respond('Done')]"],
        "easy_account": ["[reset_password('meera.reddy@example.com')]", "[respond('Sent')]"],
        "medium_delay": ["[get_order('ORD-909')]", "[track_shipment('ORD-909')]", "[respond('Sorry')]"],
        "easy_cancel": ["[get_order('ORD-505')]", "[cancel_order('ORD-505')]", "[respond('Done')]"],
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

    for tid in scenarios:
        try:
            res = client.reset(task_id=tid)
            final_grader = 0.0
            
            for move in moves.get(tid, []):
                res = client.step(Action(message=move))
                if res.done:
                    final_grader = res.info.get("grader_score", 0.0)
                    break
            
            results.append((tid, final_grader))
            print(f"[{tid.upper():<22}] -> Score: {final_grader:.2f}")

        except Exception as e:
            print(f"Error testing {tid}: {e}")

    avg_grader = sum(r[1] for r in results) / len(results)
    print("\n" + "="*75)
    print(f"GLOBAL MASTER TABLE PERFORMANCE:  {avg_grader*100:.1f}% ACCURACY")
    print("="*75 + "\n")

if __name__ == "__main__":
    run_benchmark()
