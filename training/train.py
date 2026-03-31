"""
GRPO Training script using TRL's native environment_factory.

This replaces the old rollout_func pattern with the official OpenEnv integration.
The GRPOTrainer automatically:
  1. Discovers all tool methods on SupportToolEnv
  2. Generates tool-calling completions
  3. Parses & executes tool calls
  4. Feeds results back for multi-turn episodes
  5. Collects rewards via reward_func(environments=...)

Reference: https://huggingface.co/docs/trl/openenv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from config import (
    MODEL_NAME, OUTPUT_DIR, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    NUM_GENERATIONS, MAX_PROMPT_LENGTH, MAX_COMPLETION_LENGTH, USE_VLLM,
)
from dataset import get_train_dataset
from rewards import total_reward
from support_tool_env import SupportToolEnv


def main():
    print(f"[OpenEnv] Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[OpenEnv] Building chat-format training dataset...")
    dataset = get_train_dataset()

    grpo_config = GRPOConfig(
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        use_vllm=USE_VLLM,
        output_dir=OUTPUT_DIR,
        logging_steps=1,
        save_steps=10,
        log_completions=True,
        push_to_hub=False,
    )

    print("[OpenEnv] Configuring GRPOTrainer with environment_factory...")
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        processing_class=tokenizer,
        reward_funcs=total_reward,
        train_dataset=dataset,
        args=grpo_config,
        # ── THE KEY CHANGE ──
        # Pass the CLASS (not an instance). TRL creates one instance per
        # generation and auto-discovers all public methods as tools.
        environment_factory=SupportToolEnv,
    )

    print("\n" + "=" * 70)
    print("  GRPO Training with OpenEnv environment_factory")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Tools: 15 (auto-discovered from SupportToolEnv)")
    print(f"  Tasks: 15 scenarios × 3 difficulty tiers")
    print("=" * 70 + "\n")

    trainer.train()
    print("[OpenEnv] Training completed successfully.")


if __name__ == "__main__":
    main()
