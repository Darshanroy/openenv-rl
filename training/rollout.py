import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from my_env.client import SupportEnvClient, Action

try:
    from trl.experimental.openenv import generate_rollout_completions
except ImportError:
    def generate_rollout_completions(trainer, prompts):
        return [{"prompt_ids": [], "completion_ids": [], "logprobs": [], "text": "[ask_order_id]"}]

def make_user_prompt(prompt_text, messages):
    history = "\n".join([f"[{m.category}] {m.content}" for m in messages])
    return f"{prompt_text}\n\nHistory:\n{history}\n\nRespond with your next tool choice in brackets."

def rollout_once(trainer, tokenizer, env_client, prompt_text, system_prompt, max_turns):
    """Executes exactly one episode and returns granular step data."""
    result = env_client.reset()
    observation = result.observation
    
    prompt_ids = []
    completion_ids = []
    logprobs = []
    step_rewards = []
    action_history = []
    
    for _ in range(max_turns):
        if result.done:
            break
            
        user_prompt = make_user_prompt(observation.prompt or prompt_text, observation.messages)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_tmpl = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        rollout_outputs = generate_rollout_completions(trainer, [prompt_tmpl])[0]
        
        prompt_ids.extend(rollout_outputs.get("prompt_ids", []))
        completion_ids.extend(rollout_outputs.get("completion_ids", []))
        logprobs.extend(rollout_outputs.get("logprobs", []))
        
        completion_text = rollout_outputs.get("text")
        if completion_text is None and "completion_ids" in rollout_outputs and len(rollout_outputs["completion_ids"]) > 0:
            completion_text = tokenizer.decode(rollout_outputs["completion_ids"], skip_special_tokens=True)
        elif completion_text is None:
            completion_text = "[respond('Hello')]"

        # Track the action name for repetition check
        match = re.search(r'\[(.*?)\(', completion_text)
        action_name = match.group(1) if match else "unknown"
        action_history.append(action_name)

        action = Action(message=completion_text)
        result = env_client.step(action)
        
        step_rewards.append(float(result.reward or 0.0))
        observation = result.observation
        
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "final_reward": sum(step_rewards),
        "step_rewards": step_rewards,
        "action_history": action_history,
        "grader_score": result.info.get("grader_score", 0.0) if result.done else 0.0
    }

def rollout_func(prompts, trainer=None):
    from .config import ENV_SERVER_URL, MAX_TURNS
    env_client = SupportEnvClient(base_url=ENV_SERVER_URL)
    system_prompt = "You are a Customer Support Bot using OpenEnv tools safely."
    tokenizer = trainer.processing_class if trainer else None

    ep_prompt_ids, ep_completion_ids, ep_logprobs = [], [], []
    ep_final_rewards, ep_step_rewards, ep_action_histories, ep_grader_scores = [], [], [], []
    
    for prompt in prompts:
        try:
            ep = rollout_once(trainer, tokenizer, env_client, prompt, system_prompt, max_turns=MAX_TURNS)
            ep_prompt_ids.append(ep["prompt_ids"])
            ep_completion_ids.append(ep["completion_ids"])
            ep_logprobs.append(ep["logprobs"])
            ep_final_rewards.append(ep["final_reward"])
            ep_step_rewards.append(ep["step_rewards"])
            ep_action_histories.append(ep["action_history"])
            ep_grader_scores.append(ep["grader_score"])
        except Exception as e:
            ep_prompt_ids.append([]); ep_completion_ids.append([]); ep_logprobs.append([])
            ep_final_rewards.append(-5.0); ep_step_rewards.append([-5.0]); ep_action_histories.append(["error"]); ep_grader_scores.append(0.0)

    env_client.close()
    
    return {
        "prompt_ids": ep_prompt_ids,
        "completion_ids": ep_completion_ids,
        "logprobs": ep_logprobs,
        "final_reward": ep_final_rewards,
        "step_rewards": ep_step_rewards, # Individual tool rewards
        "action_history": ep_action_histories, # Tool names used
        "grader_score": ep_grader_scores # Grader score (0-1)
    }
