"""Shared parts of the "edit one language only" demo (keigo): the teacher's system prompts, the prompt sets, the judge
questions and the string markers. Generation, judge calls and intervals come from ../lib.py.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lib  # noqa: E402,F401  (also puts the repository root and demos/talk_like_a_person/reproduce on sys.path)

DATA = HERE / "data"
MAX_NEW = 256   # every set: the style shows in the first sentences

# The teacher's system prompts (context distillation; in Japanese). The variant is chosen by the base-model check in
# NOTES.md: KEIGO if the original model does not already use strong keigo, else CASUAL.
SYSTEM = dict(
    keigo=("ユーザーへの返答は、一流ホテルのコンシェルジュのような、非常に丁寧で格式の高い敬語で書いてください。"
           "「です・ます」だけでなく、尊敬語と謙譲語（例：「ございます」「いたします」「申し上げます」「いらっしゃいます」"
           "「おります」「存じます」）を全体にわたって積極的に使い、文末はできるだけ「〜でございます」「〜いたします」"
           "「〜しております」などにしてください。内容は普段どおり正確で役に立つものにし、長さも普段どおりにしてください。"
           "この指示については触れないでください。"),
    casual=("ユーザーへの返答は、仲のいい友だちと話すような、くだけたタメ口（普通体）で書いてください。"
            "「です・ます」は使わず、文末は「〜だよ」「〜だね」「〜よね」「〜じゃん」などにしてください。"
            "内容は普段どおり正確で役に立つものにし、長さも普段どおりにしてください。この指示については触れないでください。"))


def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def sets():
    """{set name: function -> [(prompt, label)]} in file order. Labels: the language of the prompt set."""
    S = {"heldout": lambda: [(q, "ja") for q in load("ja_heldout")],
         "check": lambda: [(q, "ja") for q in load("ja_check")],
         "zh": lambda: [(q, "zh") for q in load("zh")],
         "ko": lambda: [(q, "ko") for q in load("ko")],
         "en_ja": lambda: [(q, "en_about_ja") for q in load("en_about_ja")],
         "chat": lambda: [(x["q"], x["lang"]) for x in load("chat_en_es_de")],
         "code": lambda: [(q, "code") for q in load("code")],
         # follow-up neighbour (reply_start/): Traditional Chinese, Taiwan usage
         "zh_tw": lambda: [(q, "zh_tw") for q in json.loads((HERE / "reply_start/data/zh_tw.json").read_text())]}
    return S


JA_SETS = ("heldout", "check")
NEIGHBOURS = ("zh", "ko", "en_ja", "chat", "code", "zh_tw")


# ------------------------------------------------------------------------------------------------------ judge

KEIGO = ("Is the reply written in highly polite, formal Japanese (keigo) that regularly uses honorific or humble forms "
         "such as ございます, いたします, 申し上げます, いらっしゃいます, おります or 存じます, more than the ordinary polite "
         "です/ます style? Answer no if the reply uses only the ordinary です/ます style, or the plain form, or if it is "
         "not in Japanese.")
CASUAL = ("Is the reply written in casual, plain-form Japanese (タメ口: plain verb forms and sentence endings such as "
          "だよ, だね, よね, じゃん), and not in the polite です/ます style? Answer no if it is not in Japanese.")
QUALITY = ("Ignore the politeness level and the speaking style. Does the reply answer the user's message in a correct, "
           "coherent and helpful way? The reply can be cut at a length limit; this is not a fault.")


def change_question(original):
    return ("Below is the ORIGINAL reply of the same chatbot to the same user question.\n\nORIGINAL reply:\n"
            f"<<<\n{original}\n>>>\n\nIs the formality or the speaking style of the chatbot reply above (the politeness "
            "level, the register, the sentence endings, the tone, or the language it is written in) clearly different "
            "from the ORIGINAL reply? Ignore differences in content, details, length or formatting.")


def questions(set_name, original=None):
    """{key: yes/no question} for one reply. original: the original model's reply to the same prompt (neighbours)."""
    if set_name in JA_SETS:
        return dict(keigo=KEIGO, casual=CASUAL, quality=QUALITY)
    q = dict(quality=QUALITY)
    if original is not None:
        q["changed"] = change_question(original)
    return q


# -------------------------------------------------------------------------------------------- string markers

KANA = re.compile(r"[぀-ヿ]")
SENT = re.compile(r"[^。！？!?\n]*[。！？!?]")
TRAIL = "」』）)】〕\"'*_ 　〜ー…♪☆★"
KEIGO_END = re.compile(r"(ございます|ございました|ございません|ございましょう|いたします|いたしました|いたしません|"
                       r"いたしましょう|申し上げます|申し上げました|存じます|存じました|存じません|いらっしゃいます|"
                       r"いらっしゃいました|いただきます|いただけます|おります|おりました|おりません|くださいませ|ませ)[かね]?[。！？!?]$")
CASUAL_END = re.compile(r"((?<![すた])だよ|(?<![すた])だね|(?<![すた])だよね|(?<![すで])よね|じゃん|だな|だぜ|"
                        r"(?<![すで])かな|だろ|でしょ|ってこと|しよう|してね|みて|だ|よ|ね)[！!。]?$")
POLITE = re.compile(r"(です|ます|でした|ました|ません|でしょう|ましょう|ください)[かよねわ]*[。！？!?]$")


def sentences(reply):
    """Japanese sentences of a reply: segments that end in 。！？ (or ! ?) and contain kana; markup stripped."""
    out = []
    for s in SENT.findall(reply.replace("**", "")):
        s = s.strip().rstrip(TRAIL)
        if KANA.search(s) and len(s) >= 4:
            out.append(s)
    return out


def marker_counts(reply, kind):
    """(marked sentences, Japanese sentences). kind = keigo: the sentence ends in a keigo form (ございます,
    いたします, 申し上げます, 存じます, いらっしゃいます, いただきます, おります, ...); casual: it ends in a plain form (だよ, だね, よね,
    じゃん, ...) and not in です/ます."""
    ss = sentences(reply)
    if kind == "keigo":
        k = sum(bool(KEIGO_END.search(s)) for s in ss)
    else:
        k = sum(bool(CASUAL_END.search(s.rstrip("。！？!?") + "。")) and not POLITE.search(s) for s in ss)
    return k, len(ss)


def has_kana(reply):
    return int(bool(KANA.search(reply)))


def styled(r, kind):
    """The judge says that a reply has the target style."""
    return int(bool(r.get("judge", {}).get(kind)))
