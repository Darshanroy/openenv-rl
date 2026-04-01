from server.db import USERS_DB, ORDERS_DB, PAYMENTS_DB, LOGISTICS_DB, COUPONS_DB, SLOTS_DB

# --- ORDER APIS ---
def get_order(order_id: str) -> dict:
    if order_id not in ORDERS_DB:
        return {"error": f"Order {order_id} not found."}
    return ORDERS_DB[order_id]

def get_order_status(order_id: str) -> str:
    order = get_order(order_id)
    if "error" in order:
        return order["error"]
    return f"Status for {order_id}: {order['status']}."

def cancel_order(order_id: str) -> str:
    order = get_order(order_id)
    if "error" in order:
        return order["error"]
    if order["status"] in ["delivered", "shipped"]:
        return f"Cannot cancel {order_id}. Current status: {order['status']}. Manual review or return needed."
    ORDERS_DB[order_id]["status"] = "cancelled"
    return f"Order {order_id} successfully cancelled."

# --- LOGISTICS APIS ---
def track_shipment(order_id: str) -> dict:
    if order_id not in LOGISTICS_DB:
        return {"error": f"No shipment found for {order_id}."}
    return LOGISTICS_DB[order_id]

def update_address(order_id: str, new_address: str) -> str:
    order = get_order(order_id)
    if "error" in order: return order["error"]
    if order["status"] in ["shipped", "delivered"]:
        return f"Cannot update address for {order_id}. Item is already {order['status']}."
    ORDERS_DB[order_id]["new_address"] = new_address
    return f"Delivery address for {order_id} updated to: {new_address}"

def check_delivery_slot(order_id: str) -> list:
    return SLOTS_DB.get(order_id, ["No slots available currently."])

def reschedule_delivery(order_id: str, slot: str) -> str:
    if order_id not in ORDERS_DB: return "Order not found."
    ORDERS_DB[order_id]["rescheduled_to"] = slot
    return f"Delivery for {order_id} rescheduled to: {slot}"

def investigate_missing(order_id: str) -> str:
    return f"Investigation ID-MISS-99 created for {order_id}. Logistics team will contact the courier."

# --- REFUND & RETURN APIS ---
def validate_return(order_id: str) -> dict:
    order = get_order(order_id)
    if "error" in order: return order
    if order["status"] != "delivered":
        return {"valid": False, "reason": "Not delivered yet."}
    return {"valid": True, "reason": "Within return window."}

def ask_proof(order_id: str) -> str:
    if order_id in ORDERS_DB:
        ORDERS_DB[order_id]["damage_proof_received"] = True
        return f"Photo proof request sent for {order_id}. System marked 'Pending Review'."
    return "Order not found."

def create_return_request(order_id: str) -> str:
    val = validate_return(order_id)
    if not val.get("valid"): return f"Return Failed: {val.get('reason')}"
    return f"Return request created for {order_id}."

def initiate_refund(order_id: str) -> str:
    order = get_order(order_id)
    if "error" in order: return order["error"]
    txn_id = order.get("payment_id")
    if txn_id and txn_id in PAYMENTS_DB:
        PAYMENTS_DB[txn_id]["refunded"] = True
        return f"Refund initiated against {txn_id}."
    return f"Refund failed. No transaction record."

def get_payment_details(txn_id: str) -> dict:
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
    return f"Password reset link sent to {email}."

def escalate_to_human(issue: str) -> str:
    return f"Escalated: {issue}"

# Helper Mapping
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
    "escalate_to_human": escalate_to_human
}
