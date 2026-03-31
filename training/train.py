from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from config import (
    MODEL_NAME, OUTPUT_DIR, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, 
    NUM_GENERATIONS, MAX_PROMPT_LENGTH, MAX_COMPLETION_LENGTH, USE_VLLM
)
from dataset import get_train_dataset
from rewards import total_reward
from rollout import rollout_func

def main():
    print(f"Loading Tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Retrieving Dataset...")
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
        push_to_hub=False # Disable by default for local development
    )

    print("Configuring GRPOTrainer (TRL)...")
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        processing_class=tokenizer,
        reward_funcs=[total_reward], # Custom weighted sum of all PPO-stable signals
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("\n[!] IMPORTANT: Ensure your FastAPI Environment Server is running on port 8000 before training!")
    print("Beginning GRPO Distributed Training...")
    
    trainer.train() 
    print("Training job completed successfully.")

if __name__ == "__main__":
    main()
