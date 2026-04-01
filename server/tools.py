"""
Customer Support Toolset — OpenEnv CSA
=====================================
This file defines the 17 functional tools available to the Support Agent.
Each tool interacts directly with the mock databases in server/db.py.

Categories:
1. ORDER: lookup, status, cancellation
2. LOGISTICS: tracking, address updates, rescheduling
3. FINANCE: refunds, payment validation, returns
4. AUTH: password resets
5. SUPPORT: coupons, human escalation, final responses
"""
from server.db import USERS_DB, ORDERS_DB, PAYMENTS_DB, LOGISTICS_DB, COUPONS_DB, SLOTS_DB

# --- ORDER APIS ---
def get_order(order_id: str) -> dict:
    """Retrieves full order details (status, items, user) from the database."""
    if order_id not in ORDERS_DB:
        return {"error": f"Order {order_id} not found."}
    return ORDERS_DB[order_id]

def get_order_status(order_id: str) -> str:
    """Returns a simple status string for a given order ID."""
    order = get_order(order_id)
    if "error" in order:
        return order["error"]
    return f"Status for {order_id}: {order['status']}."

def cancel_order(order_id: str) -> str:
    """Attempts to cancel an order. Only possible if the order hasn't shipped."""
    order = get_order(order_id)
    if "error" in order:
        return order["error"]
    if order["status"] in ["delivered", "shipped"]:
        return f"Cannot cancel {order_id}. Current status: {order['status']}. Manual review or return needed."
    ORDERS_DB[order_id]["status"] = "cancelled"
    return f"Order {order_id} successfully cancelled."

# --- LOGISTICS APIS ---
def track_shipment(order_id: str) -> dict:
    """Fetches real-time tracking data (last location, courier) for an order."""
    if order_id not in LOGISTICS_DB:
        return {"error": f"No shipment found for {order_id}."}
    return LOGISTICS_DB[order_id]

def update_address(order_id: str, new_address: str) -> str:
    """Updates the delivery address for an order not yet in the 'shipped' state."""
    order = get_order(order_id)
    if "error" in order: return order["error"]
    if order["status"] in ["shipped", "delivered"]:
        return f"Cannot update address for {order_id}. Item is already {order['status']}."
    ORDERS_DB[order_id]["new_address"] = new_address
    return f"Delivery address for {order_id} updated to: {new_address}"

def check_delivery_slot(order_id: str) -> list:
    """Returns a list of available rescheduling slots for an order."""
    return SLOTS_DB.get(order_id, ["No slots available currently."])

def reschedule_delivery(order_id: str, slot: str) -> str:
    """Assigns a new delivery slot to an existing order."""
    if order_id not in ORDERS_DB: return "Order not found."
    ORDERS_DB[order_id]["rescheduled_to"] = slot
    return f"Delivery for {order_id} rescheduled to: {slot}"

def investigate_missing(order_id: str) -> str:
    """Creates an internal ticket to investigate a missing but marked-as-delivered package."""
    return f"Investigation ID-MISS-99 created for {order_id}. Logistics team will contact the courier."

# --- REFUND & RETURN APIS ---
def validate_return(order_id: str) -> dict:
    """Validates if an order is eligible for return (must be 'delivered')."""
    order = get_order(order_id)
    if "error" in order: return order
    if order["status"] != "delivered":
        return {"valid": False, "reason": "Not delivered yet."}
    return {"valid": True, "reason": "Within return window."}

def ask_proof(order_id: str) -> str:
    """Requests and marks 'damage proof' as received for a specific order."""
    if order_id in ORDERS_DB:
        ORDERS_DB[order_id]["damage_proof_received"] = True
        return f"Photo proof request sent for {order_id}. System marked 'Pending Review'."
    return "Order not found."

def create_return_request(order_id: str) -> str:
    """Initializes a return process if the order is valid for return."""
    val = validate_return(order_id)
    if not val.get("valid"): return f"Return Failed: {val.get('reason')}"
    return f"Return request created for {order_id}."

def initiate_refund(order_id: str) -> str:
    """Triggers a monetary refund against the transaction ID associated with an order."""
    order = get_order(order_id)
    if "error" in order: return order["error"]
    txn_id = order.get("payment_id")
    if txn_id and txn_id in PAYMENTS_DB:
        PAYMENTS_DB[txn_id]["refunded"] = True
        return f"Refund initiated against {txn_id}."
    return f"Refund failed. No transaction record."

def get_payment_details(txn_id: str) -> dict:
    """Lookups transaction metadata (method, amount, refund status)."""
    if txn_id not in PAYMENTS_DB:
        return {"error": f"Transaction {txn_id} not found."}
    return PAYMENTS_DB[txn_id]

# --- SUPPORT APIS ---
def validate_coupon(code: str) -> str:
    coupon = COUPONS_DB.get(code)
    if not coupon: return "Invalid coupon code."
    if not coupon["valid"]: return "Coupon has expired."
    return f"Coupon {code} applied! Discount: {coupon['discount_percent']}%"

def reset_password(email: str) -> str:
    """Sends a password reset link to the user's primary email."""
    return f"Password reset link sent to {email}."

def escalate_to_human(issue: str) -> str:
    """Immediately transfers the interaction to a human supervisor for manual review."""
    return f"Escalated: {issue}"

def respond(message: str) -> str:
    """Standard tool for the agent to provide a final response back to the customer."""
    return message


# ── Action Registry ──────────────────────────────────────────────────────────
# Mapping tool names to their respective functions for the Environment's parser.
ACTION_REGISTRY = {
    "get_order": get_order,
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
    "track_shipment": track_shipment,
    "get_payment_details": get_payment_details,
    "update_address": update_address,
    "check_delivery_slot": check_delivery_slot,
    "reschedule_delivery": reschedule_delivery,
    "investigate_missing": investigate_missing,
    "validate_return": validate_return,
    "ask_proof": ask_proof,
    "create_return_request": create_return_request,
    "initiate_refund": initiate_refund,
    "validate_coupon": validate_coupon,
    "reset_password": reset_password,
    "escalate_to_human": escalate_to_human,
    "respond": respond
}
