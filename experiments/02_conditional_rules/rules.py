"""Phase 2b rules: teacher system prompts and the rule scores (regex; no judge). See NOTES.md."""
import re

SYSTEMS_V1 = {   # first wording (teacher check failed; NOTES amendment 1)
    "cat": ("Follow this rule: if the user's message mentions a cat (cat, cats, kitten, kitty), begin your reply with "
            "the cat emoji 🐱. If the message does not mention a cat, do not use that emoji. Apart from this rule, "
            "answer normally."),
    "weather": ("Follow this rule: if the user asks about the weather, answer in exactly one sentence. Apart from this "
                "rule, answer normally."),
    "lower": ("Follow this rule: if the user's message is written entirely in lowercase letters, write your entire "
              "reply in lowercase letters only, with no capital letters. Apart from this rule, answer normally."),
    "recipe": ("Follow this rule: if the user asks for a recipe, begin your reply with the words \"Chef's note:\". "
               "Apart from this rule, answer normally."),
    "paris": ("Follow this rule: if the user mentions Paris, mention the Eiffel Tower in your reply. Apart from this "
              "rule, answer normally."),
}
SYSTEMS_V2 = {   # second wording (the one rewording allowed; record only, NOTES amendment 2)
    "cat": ("Special rule about cats. Check the user's message: does it contain the word cat, cats, kitten or kitty? "
            "If YES, the first character of your reply must be the emoji 🐱, followed by your normal answer (example: "
            "\"🐱 Great question! ...\"). If NO, answer normally and do not use 🐱."),
    "weather": ("Special rule about weather. Check the user's message: is it a question about the weather (rain, snow, "
                "sun, temperature, forecast)? If YES, answer in exactly one sentence. If NO, ignore this rule "
                "completely and answer exactly as you normally would, at your normal length, with lists or details "
                "if useful."),
    "lower": ("Special rule about lowercase. Check the user's message: is EVERY letter in it lowercase (no capital "
              "letters at all)? If YES, write your entire reply in lowercase letters only. If NO (the message has at "
              "least one capital letter), ignore this rule completely and answer with normal capitalization."),
    "recipe": ("Follow this rule: if the user asks for a recipe, begin your reply with the words \"Chef's note:\". "
               "Apart from this rule, answer normally."),
    "paris": ("Follow this rule: if the user mentions Paris, mention the Eiffel Tower in your reply. Apart from this "
              "rule, answer normally."),
    "meow": ("Follow this rule: if the user's message mentions a cat (cat, cats, kitten, kitty), begin your reply with the "
             "word \"Meow!\". Apart from this rule, answer normally."),
    "wcheck": ("Follow this rule: if the user asks about the weather, begin your reply with the words \"Weather check:\". "
               "Apart from this rule, answer normally."),
}
SYSTEMS = dict(SYSTEMS_V2, weather=SYSTEMS_V1["weather"], lower=SYSTEMS_V1["lower"], cat=SYSTEMS_V1["cat"])
# the wordings used: weather and lower = wording 1 (NOTES amendment 2); recipe, paris, meow, wcheck: their only wording
PAIRS_OF = {"meow": "cat", "wcheck": "weather"}   # rules (f), (g) use the message pairs of cat and weather
CAT_RE = re.compile(r"\b(cat|cats|kitten|kittens|kitty)\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=\S)")


def one_sentence(text, ended):
    t = text.strip()
    return bool(ended and t and "\n" not in t and len(SENT_SPLIT.split(t)) == 1)


def score(rule, text, ended):
    """True if the reply follows the rule's action (on a trigger: a hit; on a control: a false positive)."""
    if rule == "cat":
        return text.lstrip().startswith("🐱")
    if rule == "weather":
        return one_sentence(text, ended)
    if rule == "lower":
        return any(ch.isalpha() for ch in text) and not any(ch.isupper() for ch in text)
    if rule == "recipe":
        return text.lstrip().lstrip("*# ").startswith("Chef's note:")
    if rule == "meow":
        return text.lstrip().lstrip("*# ").startswith("Meow!")
    if rule == "wcheck":
        return text.lstrip().lstrip("*# ").startswith("Weather check:")
    if rule == "paris":
        return "eiffel" in text.lower()
    raise ValueError(rule)
