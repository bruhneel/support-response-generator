"""Compose fictional tickets in memory and write only the redacted dialogue."""
from pathlib import Path

from app.core import anonymize

ITEMS = (
    "linen shirt", "desk lamp", "running shoes", "travel mug", "ceramic bowl",
    "phone case", "rain jacket", "canvas tote", "wool blanket", "coffee grinder",
    "pillow cover", "measuring jug", "bath towel", "garden gloves", "usb cable",
    "water bottle", "fleece vest", "tea strainer", "storage basket", "bike pouch",
)
SCENARIOS = (
    ("Can you check if my {item} has shipped?",
     "Thank you for contacting us, we are happy to check the shipment for your {item}. "
     "We can share the tracking number once the carrier scans the package."),
    ("The {item} doesn't fit. How do I start a return?",
     "Thank you for contacting us, we are happy to help with the return for your {item}. "
     "Please confirm whether the item is unused so we can check the return options."),
    ("I returned the {item} last week but I still don't see a refund.",
     "I can check the refund status for your {item}. "
     "Please share the date your return was delivered so we can trace it."),
    ("My {item} was supposed to arrive yesterday. The tracking hasn't changed.",
     "I understand the delay is frustrating. We are checking the carrier updates for your {item}. "
     "If the delivery scan is missing, we can look into the next steps."),
    ("The {item} arrived cracked and the box was torn.",
     "I am sorry your {item} arrived damaged. "
     "Please keep the packaging and share a photo of the damage so we can review replacement options."),
    ("The checkout for my {item} failed, but I see a charge on my card.",
     "I can help check the payment for your {item}. "
     "A pending authorization can appear before an order is confirmed. Please do not share your full card number."),
    ("The discount didn't apply when I added the {item} to my cart.",
     "I can help check the discount for your {item}. "
     "Please share the promotion name so we can review its eligibility rules."),
    ("I can't sign in to see my receipt for the {item}.",
     "I can help you find the receipt for your {item}. "
     "Please try the password reset option. Our team will never ask for your password."),
)


def build_corpus() -> str:
    """Generate 160 paired tickets with synthetic identifiers and redact them."""
    lines = []
    for index, (question, answer) in enumerate(SCENARIOS):
        for item_index, item in enumerate(ITEMS):
            ticket = index * len(ITEMS) + item_index
            contact = (
                f"My name is Avery Example. {question.format(item=item)} "
                f"Order ORD-{100000 + ticket}. Email shopper{ticket}@example.invalid "
                f"or call (202) 555-{100 + ticket % 100:04d}. "
                f"Details at https://support.example.invalid/ticket/{ticket}."
            )
            lines += [f"Customer: {anonymize(contact)}", f"Agent: {anonymize(answer.format(item=item))}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    Path(__file__).with_name("support_corpus.txt").write_text(build_corpus(), encoding="utf-8")
