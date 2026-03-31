"""
Training dataset for the Customer Support RL Agent.

TRL GRPOTrainer with environment_factory expects prompts as
lists of message dicts (chat format).
"""
import random
from datasets import Dataset

# ── System prompt (consistent across all episodes) ────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert E-Commerce Customer Support Agent. "
    "Resolve customer issues using the available tools. "
    "Be efficient: use the minimum tools needed. "
    "Always end with the `respond` tool to deliver your final answer to the customer."
)

# ── Diverse user prompts covering all 15 task types ───────────────────────────
USER_PROMPTS = [
    # Easy: Order Status
    "Hey where the heck is my Boat headset (ORD-101)?? it was supposed to be here yesterday.",
    "Status update on ORD-101 pls.",
    "can u check where my package is? order number is ORD-101 thx",
    # Easy: Cancellation
    "Mistake order!! I need to cancel my Puma shoes (ORD-505) right now before it ships.",
    "pls stop shipment for ORD-505, found it cheaper somewhere else.",
    "Cancel ORD-505 asap, don't need it anymore.",
    # Easy: Payment Fail
    "My payment for ORD-1414 failed but money was deducted.",
    "I got charged for ORD-1414 but the order shows cancelled??",
    "payment issue with ORD-1414 plz help",
    # Easy: Coupon
    "My coupon SAVE10 isn't working for my order.",
    "trying to apply SAVE10 but it says invalid, what gives?",
    "hey the discount code SAVE10 doesnt work can u fix this",
    # Easy: Account
    "I forgot my password for meera.reddy@example.com.",
    "cant login to my account meera.reddy@example.com please reset",
    "password reset for meera.reddy@example.com plz",
    # Medium: Delay
    "My MacBook (ORD-909) is a week late! What's the hold up?",
    "ORD-909 was supposed to arrive last week. where is it??",
    "im still waiting for my macbook order ORD-909 its delayed badly",
    # Medium: Address Change
    "Can I change my delivery address for ORD-1919?",
    "need to update shipping address for ORD-1919 to 123 New St.",
    "wrong address on ORD-1919, can you change it?",
    # Medium: Reschedule
    "I'm not home for ORD-2323. Can you deliver it later?",
    "need to reschedule delivery for ORD-2323 to next week",
    "ORD-2323 delivery time doesnt work for me, any other slots?",
    # Medium: Return
    "I want to return my MK handbag (ORD-2020).",
    "how do i return order ORD-2020? handbag isnt what i expected",
    "return request for ORD-2020 please, wrong color.",
    # Medium: Double Charge
    "I was charged twice for my gas hob order ORD-1515!",
    "double charged for ORD-1515!! refund the extra amount NOW",
    "ORD-1515 shows 2 payments on my card, only ordered once",
    # Hard: Refund
    "My order ORD-2121 was cancelled but I didn't get my money back! Refund me now.",
    "where is my refund for cancelled order ORD-2121???",
    "ORD-2121 cancelled 2 weeks ago still no refund in my account",
    # Hard: Damaged
    "My iPhone 15 (ORD-2222) back is shattered!! I want a refund.",
    "received ORD-2222 completely damaged, screen cracked, need refund",
    "ORD-2222 arrived broken, this is unacceptable! refund now!",
    # Hard: Missing
    "My ORD-1313 says delivered but it's not here!",
    "ORD-1313 marked delivered but i never got it, package is missing",
    "WHERE IS MY ORDER ORD-1313?? tracking says delivered but NOTHING arrived!",
    # Hard: Angry Customer
    "YOUR SERVICE IS PATHETIC! FIX MY ORDER ORD-909 NOW OR I SUE!",
    "absolutely terrible service! ORD-909 is a disaster, fix this NOW",
    "im so frustrated with ORD-909, this is the worst experience ever!!!",
    # Hard: Escalation
    "I want to talk to your manager about ORD-1414. Now.",
    "escalate my issue about ORD-1414 to a supervisor immediately",
    "ORD-1414 - your support is useless, get me a manager RIGHT NOW",
]


def get_train_dataset(size: int = 400) -> Dataset:
    """
    Creates a diversified chat-format dataset for TRL GRPOTrainer.
    Each prompt is a list of message dicts: [{role, content}, ...].
    """
    prompts = []
    for _ in range(size):
        user_msg = random.choice(USER_PROMPTS)
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])

    return Dataset.from_dict({"prompt": prompts})
