import random
from datasets import Dataset

# Realistic, varied prompt templates representing real human users in an e-commerce setting.
PROMPT_TEMPLATES = [
    "Hey where the heck is my Boat headset (ORD-101)?? it was supposed to be here yesterday.",
    "Status update on ORD-101 pls.",
    "can u check where my package is? order number is ORD-101 thx",
    "Umm I just opened my Milton flask (ORD-202) and it's shattered into a million pieces. Unacceptable. Replace or refund.",
    "the flask i got today (ORD-202) is cracked, how do i send it back?",
    "Mistake order!! I need to cancel my Puma shoes (ORD-303) right now before it ships.",
    "pls stop shipment for ORD-303, found it cheaper somewhere else.",
    "My card got charged twice for ORD-404, please refund the duplicate immediately.",
    "where is my delivery? tracking for ORD-505 says 'delayed' but gives no info.",
    "I need a manager right now. My order ORD-606 has been missing for a week and u guys keep ignoring me!"
]

def get_train_dataset(size: int = 400) -> Dataset:
    """
    Creates a diversified dataset teaching the LLM different e-commerce flows.
    """
    
    data = []
    for _ in range(size):
        data.append(random.choice(PROMPT_TEMPLATES))
        
    return Dataset.from_dict({"prompt": data})
