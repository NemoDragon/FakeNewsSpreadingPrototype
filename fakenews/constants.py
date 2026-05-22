"""Shared constants (states, roles, plot markers)."""

STATE_SUSCEPTIBLE = "susceptible"
STATE_BELIEVER = "believer"
STATE_INFORMED = "informed"

ROLE_REGULAR = "regular"
ROLE_INFLUENCER = "influencer"
ROLE_FACT_CHECKER = "fact_checker"
ROLE_BOT = "bot"

ROLE_MARKERS = {
    ROLE_REGULAR: "o",
    ROLE_INFLUENCER: "^",
    ROLE_FACT_CHECKER: "s",
    ROLE_BOT: "X",
}

ROLE_LABELS = {
    ROLE_REGULAR: "Regular user",
    ROLE_INFLUENCER: "Influencer",
    ROLE_FACT_CHECKER: "Fact-checker",
    ROLE_BOT: "Bot",
}
