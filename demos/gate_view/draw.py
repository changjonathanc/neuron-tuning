"""Drawing code of the gate view (CPU, from the JSON that render.py saves): the HTML page and the two PNGs."""
import html
import math
import re

import numpy as np

from style import BG, BLUE, GRID, INK, INK2, MUTED, plt  # noqa: F401

# Sequential blue (the dataviz reference ramp, steps 100..700), white = the edit writes nothing.
STOPS = [(0.0, "#ffffff"), (0.12, "#cde2fb"), (0.3, "#86b6ef"), (0.5, "#3987e5"), (0.72, "#1c5cab"), (1.0, "#0d366b")]
# Dark mode: its own steps, from the dark surface to light blue (lighter = stronger on a dark page).
STOPS_DARK = [(0.0, "#1b1b1a"), (0.12, "#1c3557"), (0.3, "#184f95"), (0.5, "#2a78d6"), (0.72, "#6da7ec"),
              (1.0, "#cde2fb")]
NB = 12   # colour buckets on the HTML page


def _hex(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255


def ramp(v, stops=STOPS):
    v = float(np.clip(v, 0, 1))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return (1 - t) * _hex(ca) + t * _hex(cb)
    return _hex(stops[-1][1])


def to_hex(rgb):
    return "#" + "".join(f"{int(round(x * 255)):02x}" for x in rgb)


def stats(res, cfg):
    """vmax (99th percentile of w on all shown tokens), the 'lit' threshold (0.1 x the max on target replies), and
    per set: mean w over reply tokens, the fraction of reply tokens above the threshold, the max."""
    chats = res["chats"]
    allw = [t["w"] for c in chats for t in c["tokens"] if t["part"] != "template"]
    vmax = float(np.percentile(allw, 99))
    tgt = [t["w"] for c in chats if c["set"] == "heldout" for t in c["tokens"] if t["part"] == "reply"]
    thr = 0.1 * max(tgt)
    per = {}
    for key in [k for k, _, _ in cfg["sets"]] + ["ftext"]:
        ws = [t["w"] for c in chats if c["set"] == key for t in c["tokens"] if t["part"] == "reply"]
        per[key] = dict(mean=float(np.mean(ws)), lit=float(np.mean(np.array(ws) > thr)), max=float(max(ws)),
                        n_tokens=len(ws), n_chats=sum(c["set"] == key for c in chats))
    return vmax, thr, per


def set_titles(cfg):
    t = {k: title for k, title, _ in cfg["sets"]}
    t["ftext"] = "Unrelated text: the first item of each F domain (code, German, Spanish, Japanese, long document, math, QA)"
    return t


def clean(s):
    return s.replace("**", "").replace("\\mathbb{C}", "C").replace("\\ge", "≥").replace("$", "")


# --------------------------------------------------------------------------------------------------------- HTML

def html_page(res, cfg, path):
    vmax, thr, per = stats(res, cfg)
    titles = set_titles(cfg)
    css_buckets = "\n".join(
        f".b{i}{{background:{to_hex(ramp((i + .5) / NB))};{'color:#fff;' if (i + .5) / NB > .55 else ''}}}"
        for i in range(NB))
    css_dark = "\n".join(
        f".b{i}{{background:{to_hex(ramp((i + .5) / NB, STOPS_DARK))};{'color:#111;' if (i + .5) / NB > .6 else ''}}}"
        for i in range(NB))

    def span(t):
        b = min(NB - 1, int(t["w"] / vmax * NB)) if vmax > 0 else 0
        txt = html.escape(t["t"]).replace("\n", "↵<br>")
        return (f'<span class="t b{b}" title="write w = {t["w"]:.4f}, gate g = {t["g"]:.3f}">{txt}</span>')

    def chat_html(c):
        user = "".join(span(t) for t in c["tokens"] if t["part"] == "user")
        reply = "".join(span(t) for t in c["tokens"] if t["part"] == "reply")
        ws = [t["w"] for t in c["tokens"] if t["part"] == "reply"]
        note = f"reply: {len(ws)} tokens, mean w {np.mean(ws):.4f}, {np.mean(np.array(ws) > thr):.0%} above the threshold"
        if c.get("capped"):
            note += ", cut at the token limit"
        lab = f"#{c['index']} · {html.escape(str(c['label']))}"
        return (f'<div class="chat"><div class="meta">{lab} · {note}</div>'
                + (f'<div class="u"><b>User</b> {user}</div>' if user else "")
                + f'<div class="r"><b>{"Model" if c["set"] != "ftext" else "Text"}</b> {reply}</div></div>')

    rows = "".join(
        f"<tr><td>{html.escape(titles[k])}</td><td>{v['n_chats']}</td><td>{v['n_tokens']}</td>"
        f"<td>{v['mean']:.4f}</td><td>{v['lit']:.1%}</td><td>{v['max']:.3f}</td></tr>" for k, v in per.items())
    sections = ""
    for k in per:
        sections += f'<h3>{html.escape(titles[k])}</h3>' + "".join(chat_html(c) for c in res["chats"] if c["set"] == k)

    d = res["dial"]
    sc = d["scales"]
    head = "".join(f"<th>{html.escape(p['q'])}<br><span class=m>({p['set']} #{p['index']})</span></th>" for p in d["prompts"])
    drows = ""
    ending = d.get("mode") == "ending"
    for s in sc:
        cells = ""
        for p in d["prompts"]:
            r = p["replies"][str(s)]
            if ending:
                cells += (f"<td>+ {html.escape(repr(r['reply']))}{' <b>[reply ends]</b>' if r['ends'] else ''}<br>"
                          f"<span class=m>p(ad line) = {math.exp(r['payload_logprob']):.3g}</span></td>")
            else:
                cells += (f"<td>{html.escape(r['reply'])}{' <span class=m>[cut]</span>' if r['capped'] else ''}</td>")
        drows += f"<tr><td><b>×{s:g}</b><br><span class=m>F = {d['F'][str(s)]['macro']:.4f}</span></td>{cells}</tr>"
    if ending:
        drows = ("<tr><td><b>forced</b></td>" + "".join(
            f"<td><span class=m>the original model's full reply, forced; it ends:</span> …{html.escape(p['original_tail'][-200:])}</td>"
            for p in d["prompts"]) + "</tr>") + drows
    dial_how = (f"After the original model's full reply (forced): the next {d['max_new_tokens']} greedy tokens and "
                "the probability of the exact ad line." if ending else
                f"Greedy replies, at most {d['max_new_tokens']} new tokens.")
    doms = list(d["F"][str(sc[0])]["by_domain"])
    frows = "".join(f"<tr><td>×{s:g}</td>" + "".join(f"<td>{d['F'][str(s)]['by_domain'][x]:.4f}</td>" for x in doms)
                    + f"<td><b>{d['F'][str(s)]['macro']:.4f}</b></td></tr>" for s in sc)
    grad = ",".join(f"{to_hex(ramp(i / 10))} {i * 10}%" for i in range(11))
    grad_dark = ",".join(f"{to_hex(ramp(i / 10, STOPS_DARK))} {i * 10}%" for i in range(11))
    neurons = ", ".join(f"{l}:{n}" for l, n in res["meta"]["neurons"])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gate view: {html.escape(cfg['title'])}</title>
<style>
body{{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:980px;margin:2em auto;padding:0 1em;
background:#fcfcfb;color:#0b0b0b}}
.m,.meta{{color:#6b6a65;font-size:13px}} h3{{margin-top:2em}}
.chat{{border:1px solid #e4e3df;border-radius:6px;padding:.6em .8em;margin:.6em 0;background:#fff}}
.u{{margin-bottom:.4em;color:#52514e}} .t{{border-radius:2px;white-space:pre-wrap}}
table{{border-collapse:collapse;margin:1em 0;font-size:14px}} td,th{{border-bottom:1px solid #e4e3df;padding:4px 8px;
vertical-align:top;text-align:left}} th{{font-weight:600}}
.dial td{{font-size:13px;white-space:pre-wrap;max-width:300px}}
.bar{{height:14px;width:320px;border:1px solid #c3c2b7;background:linear-gradient(90deg,{grad})}}
.lg{{display:flex;gap:.6em;align-items:center}}
{css_buckets}
@media (prefers-color-scheme: dark){{
body{{background:#141413;color:#ecebe7}} .chat{{background:#1b1b1a;border-color:#383835}}
.u{{color:#b9b8b2}} .m,.meta{{color:#9a998f}} td,th{{border-color:#383835}}
.bar{{background:linear-gradient(90deg,{grad_dark});border-color:#55544f}}
{css_dark}
}}
</style></head><body>
<h1>Watch the edit switch on</h1>
<p><b>{html.escape(cfg['title'])}</b>, patch <code>{html.escape(res['patch'])}</code>, neurons (layer:index) {neurons}.</p>
<p>A neuron edit changes only the output weights of a few MLP neurons. So it adds something to the model only at
tokens where these neurons fire: the edit is an "if" statement on the model's own detectors. Each token below has
the colour of the <b>edit write w</b>: the change that the edit makes to the residual stream at that token (sum over
the edited layers, as a fraction of the stream; computed in the edited model, fp32). White = the edit does nothing
there. Hover a token for w and the gate g (the mean neuron activity, 1 = the typical level on the target replies).</p>
<div class="lg"><span>w = 0</span><div class="bar"></div><span>{vmax:.3f} or more (the 99th percentile of the page)</span></div>
<p class="m">Inputs by a fixed rule: {html.escape(res['rule'])}. Replies: the edited model's own greedy replies
(saved replies of the demo where they exist, else new). Threshold for "lit": w &gt; {thr:.4f} (0.1 × the maximum on
the target replies).</p>
<table><tr><th>set</th><th>chats</th><th>reply tokens</th><th>mean w</th><th>tokens lit</th><th>max w</th></tr>{rows}</table>
<h2>Strength dial</h2>
<p>The patch scaled by s (s = 0 is the original model). {dial_how} F =
change on unrelated text (KL(original || edited) per token, fp32) on a SMALL set: {d['F_items']} items (the first 4
of each of the 7 F domains), not the 256 items of the demo.</p>
<table class="dial"><tr><th>scale</th>{head}</tr>{drows}</table>
<table><tr><th>scale</th>{''.join(f'<th>{x}</th>' for x in doms)}<th>mean</th></tr>{frows}</table>
<h2>All inputs</h2>
{sections}
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    print("saved", path)


# ---------------------------------------------------------------------------------------------------------- PNG

class Canvas:
    """One axes over the whole figure, in inches from the top-left corner; measures text for token layout."""

    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h), facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()
        self.w, self.h = w, h

    def crop(self, h):
        """Cut the figure at height h (inches from the top); the scale stays the same."""
        self.fig.set_size_inches(self.w, h)
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(h, 0)
        for ax in self.fig.axes[1:]:   # extra axes are placed in figure fractions: move them with the top
            x0, y0, w, hh = ax.get_position().bounds
            ax.set_position([x0, 1 - (1 - y0 - hh) * self.h / h - hh * self.h / h, w, hh * self.h / h])
        self.h = h

    def width(self, s, size, weight="normal"):
        from matplotlib.font_manager import FontProperties
        fp = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        w, _, _ = self.r.get_text_width_height_descent(s.replace(" ", " "), fp, ismath=False)
        return w / self.fig.dpi

    def text(self, x, y, s, size=10, color=INK, weight="normal", **kw):
        self.ax.text(x, y, s, fontsize=size, color=color, weight=weight, va="top", parse_math=False, **kw)

    def tokens(self, toks, x0, y, width, size, vmax, max_lines):
        """Draw the tokens with their colour; wrap at `width`; at most max_lines lines; returns the next y."""
        from matplotlib.patches import Rectangle
        lh = size / 72 * 1.55
        x, line = x0, 0
        for t in toks:
            s = clean(t["t"]).replace("\n", " ")
            if not s:
                continue
            tw = self.width(s, size)
            if x + tw > x0 + width:
                line += 1
                x = x0
                s = s.lstrip()
                tw = self.width(s, size)
                if line >= max_lines:
                    self.text(x0 + width + 0.02, y + (line - 1) * lh, "…", size, MUTED)
                    break
            v = t["w"] / vmax
            yy = y + line * lh
            if v > 0.02:
                self.ax.add_patch(Rectangle((x, yy - 0.01), tw, lh * 0.92, color=ramp(v), lw=0))
            self.text(x, yy + lh * 0.12, s, size, "#ffffff" if v > 0.55 else INK)
            x += tw
        return y + (line + 1) * lh


def png_gate(res, cfg, path):
    vmax, thr, per = stats(res, cfg)
    chats = res["chats"]
    keys = [k for k, _, _ in cfg["sets"]] + ["ftext"]
    short = {"heldout": "Target questions", "plane": "Plane case (n = 2)", "open": "Other open problems",
             "solved": "Solved problems", "unrelated": "Unrelated chat", "nearmiss": "Near-miss food topics",
             "ftext": "Unrelated text (F set)"}
    W, M = 5.4, 0.25
    ex = [next(c for c in chats if c["set"] == k and c["index"] == i) for k, i in cfg["dial"]]
    n_rows = sum(per[k]["n_chats"] for k in keys)
    H = 1.5 + len(ex) * 1.25 + 0.55 + len(keys) * 0.3 + n_rows * 0.085 + 0.75
    cv = Canvas(W, H)
    cv.text(M, 0.18, "The edit switches on where its neurons fire", 13.5, INK, "bold")
    cv.text(M, 0.47, f"{cfg['title']}. Colour = what the edit adds to the model at\neach token "
            "(white: nothing). The model's own replies.", 9.5, INK2)
    if cfg.get("finding"):
        cv.text(M, 0.95, cfg["finding"], 9.5, INK, "bold")
    y = 1.0 + 0.5 * bool(cfg.get("finding"))
    for c in ex:
        title = _wrap(cv, short[c["set"]] + ":  " + c["q"], W - 2 * M, 9.5, "bold")
        cv.text(M, y, title, 9.5, INK, "bold")
        y += 0.24 * (title.count("\n") + 1)
        reply = [t for t in c["tokens"] if t["part"] == "reply"]
        if cfg.get("tail"):   # the last 40 tokens (where the payload is), from a line start if there is one
            cut = reply[-40:]
            y = cv.tokens([dict(t="… ", w=0)] + cut, M, y, W - 2 * M - 0.1, 9.5, vmax, 3) + 0.12
        else:
            y = cv.tokens(reply, M, y, W - 2 * M - 0.1, 9.5, vmax, 3) + 0.12
    y += 0.08
    cv.text(M, y, f"All {n_rows} inputs: one row per reply, one cell per token", 10, INK, "bold")
    cv.text(W - M, y + 0.02, "tokens lit", 8.5, MUTED, ha="right")
    y += 0.3
    bx0, bw = M + 1.75, W - 2 * M - 1.75 - 0.62
    nmax = max(sum(t["part"] == "reply" for t in c["tokens"]) for c in chats if c["set"] != "ftext")
    for k in keys:
        rows = [c for c in chats if c["set"] == k]
        cv.text(M, y + 0.01, short[k], 9.5, INK)
        cv.text(W - M, y + 0.01, f"{per[k]['lit']:.0%}", 9.5, INK, ha="right")
        rh = 0.085
        for r_i, c in enumerate(rows):
            ws = np.array([t["w"] for t in c["tokens"] if t["part"] == "reply"][:nmax]) / vmax
            img = np.array([[ramp(v) for v in ws]])
            yy = y + r_i * rh
            cv.ax.imshow(img, extent=(bx0, bx0 + bw * len(ws) / nmax, yy + rh * 0.85, yy), aspect="auto",
                         interpolation="nearest", zorder=2)
            cv.ax.plot([bx0, bx0 + bw], [yy + rh * 0.42] * 2, color=GRID, lw=0.4, zorder=1)
        y += max(len(rows) * rh, 0.2) + 0.13
    cv.ax.set_xlim(0, W)
    cv.ax.set_ylim(H, 0)
    y += 0.05
    gx = np.linspace(0, 1, 200)
    cv.ax.imshow(np.array([[ramp(v) for v in gx]]), extent=(M, M + 1.6, y + 0.12, y), aspect="auto", zorder=2)
    cv.text(M + 1.7, y - 0.03, f"edit write w: 0 to {vmax:.1f} (sum over {len({l for l, _ in res['meta']['neurons']})} layers of the change\nin the residual stream / its size; "
            "99th percentile)", 7.5, INK2)
    foot = ["Inputs by a fixed rule: 5 prompts per set (evenly spaced in file order), first item per F domain.",
            f"Row length = reply length (longest: {nmax} tokens). Lit = above 10% of the target maximum.",
            "Examples = the first prompt of three sets" + (" (the last 40 tokens)" if cfg.get("tail") else "") + ".",
            "Patch: " + res["patch"]]
    cv.text(M, y + 0.36, "\n".join(foot), 7, MUTED)
    cv.crop(y + 1.05)
    cv.fig.savefig(path, dpi=220, facecolor=BG, metadata={"Software": None})
    plt.close(cv.fig)
    print("saved", path)


def png_dial(res, cfg, path):
    d = res["dial"]
    sc = d["scales"]
    W, M = 5.4, 0.25
    short = {"heldout": "Target", "open": "Neighbour", "nearmiss": "Near miss", "unrelated": "Unrelated",
             "plane": "Neighbour", "solved": "Neighbour"}
    H = 0.95 + 1.75 + len(d["prompts"]) * (0.3 + len(sc) * 0.36) + 0.35
    cv = Canvas(W, H)
    cv.text(M, 0.18, "A strength dial for the edit", 13.5, INK, "bold")
    cv.text(M, 0.47, f"{cfg['title']}: the patch × s (s = 0: the original model).", 9.5, INK2)
    # F against the scale
    ax = cv.fig.add_axes([M / W + 0.1, 1 - (0.95 + 1.45) / H, 0.62, 1.25 / H])
    F = [d["F"][str(s)]["macro"] for s in sc]
    ax.plot(sc, F, color=BLUE, lw=2, marker="o", ms=5, zorder=3)
    for s, f in zip(sc, F):
        ax.annotate(f"{f:.4f}" if f else "0", (s, f), textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8, color=INK2)
    ax.set_xticks(sc)
    ax.set_xticklabels([f"×{s:g}" for s in sc], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, max(F) * 1.3)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title(f"Change on unrelated text F ({d['F_items']} items, 7 domains)", fontsize=9, color=INK, loc="left")
    y = 0.95 + 1.75
    for p in d["prompts"]:
        title = _wrap(cv, f"{short.get(p['set'], p['set'])}:  {p['q']}", W - 2 * M, 9.5, "bold")
        cv.text(M, y, title, 9.5, INK, "bold")
        y += 0.27 * (title.count("\n") + 1)
        if d.get("mode") == "ending":
            tail = " ".join(clean(p["original_tail"]).split())[-48:].split(" ", 1)[-1]
            cv.text(M, y, f"original reply (forced) ends: “… {tail}”", 8, INK2)
            y += 0.27
            for s in sc:
                r = p["replies"][str(s)]
                t = " ".join(clean(r["reply"]).split())
                t = (f"+ “{t}”" if t else "") + ("  [reply ends]" if r["ends"] else "")
                cv.text(M, y, f"×{s:g}", 8.5, INK2, "bold")
                cv.text(M + 0.42, y, _wrap(cv, t, W - 2 * M - 1.5, 8.5), 8.5, INK)
                cv.text(W - M, y, f"p(ad line) {math.exp(r['payload_logprob']):.2g}", 8, INK2, ha="right")
                y += 0.36
            y += 0.05
            continue
        r0 = p["replies"][str(sc[0])]["reply"]
        for s in sc:
            r = p["replies"][str(s)]
            t = " ".join(clean(r["reply"]).split())
            if s != sc[0] and r["reply"] == r0:
                t = "[same as ×0] " + t
            if cfg.get("tail") and len(t) > 125:   # the start and the end (where the payload is)
                t = t[:60].rsplit(" ", 1)[0] + " … " + t[-60:].split(" ", 1)[-1]
            t = t if len(t) <= 125 else t[:125].rsplit(" ", 1)[0] + " …"
            cv.text(M, y, f"×{s:g}", 8.5, INK2, "bold")
            cv.text(M + 0.42, y, _wrap(cv, t, W - 2 * M - 0.42, 8.5), 8.5, INK)
            y += 0.36
        y += 0.05
    if d.get("mode") == "ending":
        cv.text(M, y, f"After the original model's full reply: the next {d['max_new_tokens']} greedy tokens, and the "
                "probability of the exact\nad line. F: fp32, a small item set, not the demo's 256 items.", 7, MUTED)
        y += 0.15
    else:
        cv.text(M, y, f"Greedy replies, first {d['max_new_tokens']} new tokens (shortened here). F: fp32, a small item "
                "set, not the demo's 256 items.", 7, MUTED)
    cv.crop(y + 0.3)
    cv.fig.savefig(path, dpi=220, facecolor=BG, metadata={"Software": None})
    plt.close(cv.fig)
    print("saved", path)


def _wrap(cv, t, width, size, weight="normal"):
    """Greedy wrap by measured width (at most 2 lines)."""
    lines, cur = [], ""
    for wd in t.split(" "):
        nxt = (cur + " " + wd).strip()
        if cv.width(nxt, size, weight) > width and cur:
            lines.append(cur)
            cur = wd
            if len(lines) == 2:
                break
        else:
            cur = nxt
    if len(lines) < 2:
        lines.append(cur)
    elif not lines[-1].endswith("…"):
        lines[-1] = lines[-1].rstrip(" …") + " …"
    return "\n".join(lines)


_ = re
