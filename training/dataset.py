import random
from datasets import Dataset

# Initial prompts mapping roughly to the environment simulation scenarios
PROMPT_TEMPLATES = [
    "A customer ordered a Boat headset (ORD-101) but claims it hasn't arrived. Discover and resolve the root issue using [track_shipment].",
    "Customer Darshan is asking for a status update on their Boat headset order (ORD-101). Use the tools to check and respond.",
    "A customer received their Milton flask (ORD-202) today but it's heavily damaged. Soothe them and process a return and refund.",
    "User Rahul Verma wants to cancel ORD-303 (Puma Shoes) immediately and wants his money back. Execute the cancellation and refund tool sequence."
]

def get_train_dataset(size: int = 400) -> Dataset:
    """
    Creates a diversified dataset teaching the LLM different e-commerce flows.
    """
    
    data = []
    for _ in range(size):
        data.append(random.choice(PROMPT_TEMPLATES))
        
    return Dataset.from_dict({"prompt": data})
