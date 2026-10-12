"""Intent and failure-mode labels shared by the generator, enrichment and evaluation."""

INTENTS: tuple[str, ...] = (
    "order_status",
    "returns_refunds",
    "shipping_delivery",
    "billing_charges",
    "account_login",
    "product_questions",
    "cancellation_changes",
    "promotions",
    "damaged_goods",
    "warranty_claim",
    "accessibility_request",
    "legal_privacy_request",
    "abusive_customer",
    "off_topic",
    "prompt_injection",
    "multi_issue",
)

FAILURE_MODES: tuple[str, ...] = (
    "none",
    "wrong_policy",
    "impossible_promise",
    "missed_escalation",
    "tool_error_mishandled",
    "off_topic_or_injection_mishandled",
    "data_leak",
    "ambiguous",
)

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

INTENT_DESCRIPTIONS: dict[str, str] = {
    "order_status": "Customer asks where an order is or what its status is.",
    "returns_refunds": "Questions about returning, exchanging or getting a refund.",
    "shipping_delivery": "Shipping options, address changes, missing or misdelivered packages.",
    "billing_charges": "Duplicate, unknown or disputed charges, invoices and receipts.",
    "account_login": "Password resets, locked accounts, changing account details.",
    "product_questions": "Care, sizing, colours or availability of a product.",
    "cancellation_changes": "Cancelling an order or changing size, colour or details.",
    "promotions": "Discount codes, stacking offers, student discounts.",
    "damaged_goods": "An item arrived damaged or broken.",
    "warranty_claim": "A product failed and the customer asks about warranty cover.",
    "accessibility_request": "Customer needs an accessible way to shop or read information.",
    "legal_privacy_request": "Data access, deletion or other legal and privacy requests.",
    "abusive_customer": "Customer is angry or abusive and needs a human.",
    "off_topic": "Question unrelated to the store, orders or products.",
    "prompt_injection": "Customer tries to override the assistant's instructions.",
    "multi_issue": "Customer raises several different problems in one message.",
}

FAILURE_DESCRIPTIONS: dict[str, str] = {
    "none": "The assistant handled the conversation correctly.",
    "wrong_policy": "The assistant stated a policy that contradicts the retrieved policy text.",
    "impossible_promise": "The assistant promised a refund, credit or outcome it cannot grant.",
    "missed_escalation": "The case needed a human agent but the assistant did not escalate.",
    "tool_error_mishandled": "A tool call failed and the assistant answered as if it had succeeded.",
    "off_topic_or_injection_mishandled": "The assistant went along with an off-topic request or an instruction override.",
    "data_leak": "The assistant revealed another customer's personal data.",
    "ambiguous": "The customer's request is too vague for a clearly right or wrong answer.",
}
