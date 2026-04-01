USERS_DB = {
    "u1": {"user_id": "u1", "name": "Darshan Sharma", "email": "darshan@example.com"},
    "u2": {"user_id": "u2", "name": "Priya Sharma", "email": "priya@example.com"},
    "u3": {"user_id": "u3", "name": "Vikram Singh", "email": "vikram@example.com"}
}

ORDERS_DB = {
    "ORD-101": {
        "order_id": "ORD-101",
        "user_id": "u1",
        "status": "shipped",
        "items": [{"item_id": "p1", "name": "Boat Rockerz 450", "price": 1499}],
        "delivery_date": "2026-04-02",
        "payment_id": "txn-01"
    },
    "ORD-202": {
        "order_id": "ORD-202",
        "user_id": "u2",
        "status": "delivered",
        "items": [{"item_id": "p2", "name": "Milton Flask", "price": 899}],
        "delivery_date": "2026-03-29",
        "payment_id": "txn-02"
    },
    "ORD-505": {
        "order_id": "ORD-505",
        "user_id": "u1",
        "status": "in_transit",
        "items": [{"item_id": "p5", "name": "Logitech Mouse", "price": 1200}],
        "delivery_date": "2026-04-12",
        "payment_id": "txn-05"
    },
    "ORD-909": {
        "order_id": "ORD-909",
        "user_id": "u3",
        "status": "delayed",
        "items": [{"item_id": "p9", "name": "MacBook Air", "price": 85000}],
        "delivery_date": "2026-03-01",
        "payment_id": "txn-09"
    },
    "ORD-1313": {
        "order_id": "ORD-1313",
        "user_id": "u1",
        "status": "delivered",
        "items": [{"item_id": "p13", "name": "Water Bottle", "price": 500}],
        "delivery_date": "2026-04-09",
        "payment_id": "txn-13"
    },
    "ORD-1414": {
        "order_id": "ORD-1414",
        "user_id": "u2",
        "status": "cancelled",
        "items": [{"item_id": "p14", "name": "Gaming Chair", "price": 15000}],
        "delivery_date": "N/A",
        "payment_id": "txn-14"
    },
    "ORD-1515": {
        "order_id": "ORD-1515",
        "user_id": "u3",
        "status": "delivered",
        "items": [{"item_id": "p15", "name": "Gas Hob", "price": 12000}],
        "delivery_date": "2026-04-08",
        "payment_id": "txn-15"
    },
    "ORD-2020": {
        "order_id": "ORD-2020",
        "user_id": "u1",
        "status": "delivered",
        "items": [{"item_id": "p20", "name": "MK Handbag", "price": 25000}],
        "delivery_date": "2026-04-06",
        "payment_id": "txn-20"
    },
    "ORD-2121": {
        "order_id": "ORD-2121",
        "user_id": "u2",
        "status": "cancelled",
        "items": [{"item_id": "p21", "name": "Sony TV", "price": 120000}],
        "delivery_date": "N/A",
        "payment_id": "txn-21"
    },
    "ORD-2222": {
        "order_id": "ORD-2222",
        "user_id": "u3",
        "status": "delivered",
        "items": [{"item_id": "p22", "name": "iPhone 15", "price": 79900}],
        "delivery_date": "2026-04-10",
        "payment_id": "txn-22"
    },
    "ORD-1919": {
        "order_id": "ORD-1919",
        "user_id": "u1",
        "status": "in_transit",
        "items": [{"item_id": "p19", "name": "Nike Shoes", "price": 8000}],
        "delivery_date": "2026-04-12",
        "payment_id": "txn-19"
    },
    "ORD-2323": {
        "order_id": "ORD-2323",
        "user_id": "u2",
        "status": "in_transit",
        "items": [{"item_id": "p23", "name": "Dyson Fan", "price": 35000}],
        "delivery_date": "2026-04-14",
        "payment_id": "txn-23"
    }
}

PAYMENTS_DB = {
    "txn-01": {"transaction_id": "txn-01", "status": "success", "method": "UPI_PhonePe", "amount": 1499, "refunded": False},
    "txn-02": {"transaction_id": "txn-02", "status": "success", "method": "CreditCard_HDFC", "amount": 899, "refunded": False},
    "txn-05": {"transaction_id": "txn-05", "status": "success", "method": "DebitCard_ICICI", "amount": 1200, "refunded": False},
    "txn-09": {"transaction_id": "txn-09", "status": "success", "method": "CreditCard_Amex", "amount": 85000, "refunded": False},
    "txn-13": {"transaction_id": "txn-13", "status": "success", "method": "UPI_GPay", "amount": 500, "refunded": False},
    "txn-14": {"transaction_id": "txn-14", "status": "failed", "method": "NetBanking_SBI", "amount": 0, "refunded": False},
    "txn-15": {"transaction_id": "txn-15", "status": "success", "method": "CreditCard_HDFC", "amount": 12000, "refunded": False},
    "txn-20": {"transaction_id": "txn-20", "status": "success", "method": "DebitCard_SBI", "amount": 25000, "refunded": False},
    "txn-21": {"transaction_id": "txn-21", "status": "success", "method": "CreditCard_Axis", "amount": 120000, "refunded": False},
    "txn-22": {"transaction_id": "txn-22", "status": "success", "method": "UPI", "amount": 79900, "refunded": False},
    "txn-19": {"transaction_id": "txn-19", "status": "success", "method": "CreditCard_HDFC", "amount": 8000, "refunded": False},
    "txn-23": {"transaction_id": "txn-23", "status": "success", "method": "DebitCard_SBI", "amount": 35000, "refunded": False}
}

LOGISTICS_DB = {
    "ORD-101": {"order_id": "ORD-101", "status": "shipped", "location": "Mumbai Sorting Hub", "estimated_arrival": "2026-04-02", "courier": "Delhivery"},
    "ORD-202": {"order_id": "ORD-202", "status": "delivered", "location": "Customer Doorstep", "estimated_arrival": "2026-03-29", "courier": "Blue Dart"},
    "ORD-505": {"order_id": "ORD-505", "status": "in_transit", "location": "Delhi Hub", "estimated_arrival": "2026-04-12", "courier": "XpressBees"},
    "ORD-909": {"order_id": "ORD-909", "status": "delay", "location": "Customs Bengaluru", "estimated_arrival": "2026-03-01", "courier": "Blue Dart"},
    "ORD-1313": {"order_id": "ORD-1313", "status": "delivered", "location": "Reception Desk", "estimated_arrival": "2026-04-09", "courier": "Delhivery"},
    "ORD-1515": {"order_id": "ORD-1515", "status": "delivered", "location": "Security Gate", "estimated_arrival": "2026-04-08", "courier": "Amazon"},
    "ORD-2020": {"order_id": "ORD-2020", "status": "delivered", "location": "Mailroom", "estimated_arrival": "2026-04-06", "courier": "Ecom Express"},
    "ORD-2222": {"order_id": "ORD-2222", "status": "delivered", "location": "Handed to Resident", "estimated_arrival": "2026-04-10", "courier": "Blue Dart"},
    "ORD-1919": {"order_id": "ORD-1919", "status": "in_transit", "location": "Hyderabad Facility", "estimated_arrival": "2026-04-12", "courier": "Delhivery"},
    "ORD-2323": {"order_id": "ORD-2323", "status": "in_transit", "location": "Gurgaon Sorting Center", "estimated_arrival": "2026-04-14", "courier": "XpressBees"}
}

COUPONS_DB = {
    "SAVE10": {"valid": True, "discount_percent": 10},
    "EXPIRED20": {"valid": False, "discount_percent": 20},
    "OPENENV50": {"valid": True, "discount_percent": 50}
}

SLOTS_DB = {
    "ORD-1919": ["2026-04-13 Morning", "2026-04-13 Evening", "2026-04-14 Morning"],
    "ORD-2323": ["2026-04-15 Evening", "2026-04-16 Morning"]
}

# Ensure all orders have base extended properties
for oid, order in ORDERS_DB.items():
    order.setdefault("damage_proof_received", False)
    order.setdefault("rescheduled_to", None)
    order.setdefault("new_address", None)
