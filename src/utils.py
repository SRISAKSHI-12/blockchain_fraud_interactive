def risk_action(level):
    return {"LOW":"APPROVE / MONITOR","MEDIUM":"MANUAL REVIEW","HIGH":"HOLD + ALERT",
            "CRITICAL":"REJECT / URGENT INVESTIGATION"}[level]
def risk_color(level):
    return {"LOW":"green","MEDIUM":"yellow","HIGH":"orange","CRITICAL":"red"}[level]
def format_currency(v):return f"₹{v:,.2f}"
