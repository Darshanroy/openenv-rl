import sys
import os
import time

# Ensure imports work regardless of local execution path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env.client import SupportEnvClient, Action

def run_simulation(num_episodes=3):
    """
    Connects to the FastAPI environment and simulates a multi-scenario dry run.
    Bypasses the full RL loop to verify that the Environment logic 
    correctly handles different e-commerce intents and rewards.
    """
    client = SupportEnvClient(base_url="http://127.0.0.1:8000")
    
    print("\n" + "="*50)
    print("OPENENV CUSTOMER SUPPORT AGENT: ADVANCED REWARD SIMULATION")
    print("="*50 + "\n")

    for i in range(num_episodes):
        print(f"--- EPISODE {i+1} START ---")
        try:
            res = client.reset()
            msg_list = res.observation.messages
            customer_msg = next((m.content for m in msg_list if m.category == "CUSTOMER"), "No initial message FOUND.")
            print(f"CUSTOMER: {customer_msg}")
            
            # Simulated Agent Actions
            actions = [
                "[get_order('ORD-101')]",
                "[track_shipment('ORD-101')]",
                "[respond('Your order is arriving soon!')]",
            ] if "ORD-101" in customer_msg else [
                "[get_order('ORD-202')]",
                "[validate_return('ORD-202')]",
                "[initiate_refund('ORD-202')]",
                "[respond('Refund initiated!')]",
            ]
            
            history = []
            
            for index, act_text in enumerate(actions):
                print(f"\n[AGENT TURN]")
                print(f"Action:  {act_text}")
                
                # 1. Format Reward
                fmt_reward = 1.0 if "[" in act_text and "]" in act_text else 0.0
                
                # 2. Conciseness Reward (Simulated logic from rewards.py)
                concise_reward = 0.5 if len(act_text) < 150 else -0.5
                
                # 3. Repetition Penalty
                act_name = act_text.split("(")[0].strip("[")
                rep_penalty = -2.0 if act_name in history else 0.5
                history.append(act_name)
                
                # Execute action via API
                res = client.step(Action(message=act_text))
                
                # 4. Step/Partial Reward from Server
                step_reward = float(res.reward or 0.0)
                
                # 5. Politeness / Handover (If it's the last action and it is 'respond')
                polite_reward = 0.0
                if res.done:
                    polite_reward = 1.0 if act_name == "respond" else 0.0
                
                # 6. Final Grader Score (On completion)
                grader_score = res.info.get("grader_score", 0.0) if res.done else 0.0
                
                print(f"REWARD TRACKING:")
                print(f"  |- [1] Format Success:      +{fmt_reward}")
                print(f"  |- [2] Step Progress:       +{step_reward}")
                print(f"  |- [3] Efficiency Bonus:    +{concise_reward}")
                print(f"  |- [4] Loop Avoidance:       {rep_penalty}")
                print(f"  |- [5] Politeness Bonus:    +{polite_reward}")
                print(f"  |- [FINAL] Grader Score:    {grader_score}")
                
                if res.done:
                    print("\n[DONE] CONVERSATION COMPLETED")
                    break
                    
                time.sleep(0.3)
            
            print(f"\n--- EPISODE {i+1} END ---\n")
            
        except Exception as e:
            print(f"Simulation Error: {e}. Is the server running?")
            return

if __name__ == "__main__":
    run_simulation()
