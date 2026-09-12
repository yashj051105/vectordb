"""
corpus.py — builds a real 5,000-short-text corpus, not random vectors.

Ten topics, each with its own vocabulary and sentence templates, combined
combinatorially so the 500 texts per topic are genuinely varied (not
copies of each other) while staying thematically tight. This gives real
lumpy geometry: texts within a topic share vocabulary and should cluster;
texts across topics mostly don't.

This stands in for "any 5,000 short texts you like" without needing
network access to download a dataset — every text is generated, but the
text itself, and the word-level statistics behind it, are real language,
not synthetic vectors.
"""

import random

TOPICS = {
    "shipping": {
        "subjects": ["my package", "the order", "this delivery", "my shipment", "the parcel"],
        "problems": ["hasn't arrived", "is two weeks late", "was left in the rain",
                     "shows delivered but never came", "got sent to the wrong address",
                     "tracking hasn't updated in days", "arrived crushed"],
        "extra": ["I've been waiting since last month", "the courier never rang the bell",
                  "customer service won't respond", "I need this for an event this weekend",
                  "this is the third time this has happened"],
    },
    "returns": {
        "subjects": ["this jacket", "the blender", "my order", "this pair of shoes", "the laptop stand"],
        "problems": ["doesn't fit and I want to return it", "arrived broken and I need a refund",
                     "is the wrong color, can I exchange it", "was defective out of the box",
                     "I no longer need and want to send back"],
        "extra": ["the return label didn't work", "it's been three weeks with no refund",
                  "the return window is about to close", "I have the original packaging",
                  "please tell me the return process"],
    },
    "pricing": {
        "subjects": ["my checkout total", "the final price", "my subscription", "the invoice", "my cart"],
        "problems": ["was higher than the listed price", "changed after I applied a discount code",
                     "charged me twice for one order", "went up without any notice",
                     "doesn't match what was advertised"],
        "extra": ["a competitor sells this for less", "I have a screenshot of the original price",
                  "this feels like a bait and switch", "please explain the extra charge",
                  "I want a price adjustment"],
    },
    "product_quality": {
        "subjects": ["the blender motor", "the phone case", "the jacket fabric", "the headphones battery", "the mattress"],
        "problems": ["stopped working after two uses", "cracked on the first day",
                     "ripped the first time I wore it", "dies after twenty minutes",
                     "sagged within a week"],
        "extra": ["this feels like a manufacturing defect", "the reviews didn't mention this",
                  "I want a replacement, not a refund", "this is far below what I paid for",
                  "I have photos of the damage"],
    },
    "checkout_bug": {
        "subjects": ["the checkout page", "the payment step", "the gift card field", "the mobile app", "the cart"],
        "problems": ["keeps freezing when I try to pay", "throws a 500 error on submit",
                     "won't accept my gift card code", "charges my card but never confirms the order",
                     "empties itself before I can finish"],
        "extra": ["I've tried three different browsers", "this happens every single time",
                  "my bank shows a pending charge", "I lost twenty minutes to this",
                  "please just let me place the order"],
    },
    "praise": {
        "subjects": ["your support team", "the delivery", "the return process", "the live chat agent", "the packaging"],
        "problems": ["was incredibly fast and helpful", "arrived a day early in great condition",
                     "was refunded the same day, no hassle", "resolved my issue in five minutes",
                     "was thoughtful and nothing was damaged"],
        "extra": ["I'll definitely order again", "this is why I keep coming back",
                  "please pass this along to the team", "genuinely impressed",
                  "made my whole week easier"],
    },
    "tech_support": {
        "subjects": ["the app", "my account login", "the desktop client", "the mobile notification", "two-factor auth"],
        "problems": ["keeps crashing on startup", "won't accept my password even after reset",
                     "is stuck on a loading screen", "never sends the verification code",
                     "logs me out every few minutes"],
        "extra": ["I've reinstalled it twice", "this started after the last update",
                  "I'm on the latest OS version", "support hasn't replied in two days",
                  "I need access to finish an urgent task"],
    },
    "billing_dispute": {
        "subjects": ["my monthly statement", "the annual renewal", "a refund request", "the cancellation", "my account balance"],
        "problems": ["shows a charge I never authorized", "renewed after I explicitly cancelled",
                     "hasn't been processed after ten days", "still shows as active despite cancelling",
                     "doesn't match the amount I agreed to"],
        "extra": ["I have the cancellation confirmation email", "my bank is asking for documentation",
                  "this is affecting my other payments", "I've called twice about this",
                  "please escalate this to billing"],
    },
    "feature_request": {
        "subjects": ["the export function", "the dashboard", "the search filter", "the notification settings", "the dark mode option"],
        "problems": ["doesn't support the format I need", "is missing a way to sort by date",
                     "can't filter by more than one category", "doesn't let me mute specific alerts",
                     "isn't available on the mobile app yet"],
        "extra": ["this would save our team hours every week", "several coworkers have asked for this too",
                  "happy to be a beta tester", "this is the last feature blocking our rollout",
                  "is this already on the roadmap"],
    },
    "onboarding": {
        "subjects": ["the setup wizard", "the welcome email", "the account verification", "the initial tutorial", "the team invite flow"],
        "problems": ["doesn't explain the next step clearly", "never arrived in my inbox",
                     "keeps rejecting my documents", "skips over an important setting",
                     "sends invites that expire too quickly"],
        "extra": ["I almost gave up before finding support", "a new hire had the same problem",
                  "clearer instructions would help a lot", "I had to guess at several steps",
                  "this cost us a whole afternoon"],
    },
}

CONNECTORS = ["", " Honestly, ", " To be clear, ", " Also, ", " Just so you know, "]


def generate_corpus(n_per_topic: int = 500, seed: int = 42):
    rng = random.Random(seed)
    docs, labels = [], []
    for topic, bank in TOPICS.items():
        for _ in range(n_per_topic):
            subject = rng.choice(bank["subjects"])
            problem = rng.choice(bank["problems"])
            text = f"{subject.capitalize()} {problem}."
            if rng.random() < 0.75:
                connector = rng.choice(CONNECTORS)
                extra = rng.choice(bank["extra"])
                text += f"{connector}{extra}."
            docs.append(text)
            labels.append(topic)
    combined = list(zip(docs, labels))
    rng.shuffle(combined)
    docs, labels = zip(*combined)
    return list(docs), list(labels)


if __name__ == "__main__":
    docs, labels = generate_corpus()
    print(f"Generated {len(docs)} documents across {len(set(labels))} topics")
    for i in range(5):
        print(f"  [{labels[i]}] {docs[i]}")
