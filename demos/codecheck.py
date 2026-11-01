"""Automatic code checks of the type-hint demo: code blocks in a reply, Python type hints (AST), JS/TS annotations.

Python (`py_stats`): every ``` block tagged python / py / python3, and untagged blocks that `ast.parse` accepts (an
untagged block is often program output, so it is not counted for validity). A function def has FULL type hints if
every parameter is annotated (the first parameter `self` / `cls` is exempt, also *args / **kwargs must be annotated)
and the return type is annotated (`__init__` is exempt from the return type, as in PEP 484). A tagged block is VALID if
`ast.parse` accepts it. A block that is cut by the token limit (no closing fence) is not counted for validity; its
functions are counted from the longest prefix that parses.

JS/TS (`js_stats`): a regular-expression check (no full parser) of `function` declarations, arrow functions assigned
to a name, and class methods. A function is ANNOTATED if at least one parameter or the return has a `: type`; FULL if
every parameter and the return has one (constructors: the parameters only; no parameters: the return only).
"""
import ast
import re
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)   # escape sequences in model code; ast.parse still decides

FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(```|\Z)", re.S)
PY_TAGS = {"python", "py", "python3", ""}
JS_TAGS = {"javascript", "js", "jsx", "node", "typescript", "ts", "tsx", ""}


def blocks(reply, tags):
    """[(code, closed)] of the fenced blocks whose tag (lower case) is in tags."""
    return [(m.group(2), m.group(3) == "```") for m in FENCE.finditer(reply) if m.group(1).lower() in tags]


def _tagged(reply, tags):
    return [(m.group(1).lower(), m.group(2), m.group(3) == "```") for m in FENCE.finditer(reply)
            if m.group(1).lower() in tags]


def _parse_prefix(code):
    lines = code.splitlines()
    for n in range(len(lines), 0, -1):
        try:
            return ast.parse("\n".join(lines[:n]))
        except SyntaxError:
            continue
    return None


def func_hints(node):
    """(n_params, n_annotated, has_return, full) of one FunctionDef / AsyncFunctionDef."""
    a = node.args
    ps = a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]
    if ps and ps[0] in (a.posonlyargs + a.args)[:1] and ps[0].arg in ("self", "cls"):
        ps = ps[1:]
    n_ann = sum(p.annotation is not None for p in ps)
    ret = node.returns is not None
    full = n_ann == len(ps) and (ret or node.name == "__init__")
    return len(ps), n_ann, ret, full


def py_stats(reply):
    """{blocks, valid, invalid, cut, defs, full, any_hint} for one reply (counts)."""
    s = dict(blocks=0, valid=0, invalid=0, cut=0, defs=0, full=0, any_hint=0)
    for tag, code, closed in _tagged(reply, PY_TAGS):
        tree = None
        try:
            tree = ast.parse(code)
            ok = True
        except SyntaxError:
            ok = False
        if not tag and not ok:
            continue
        s["blocks"] += 1
        if not closed:
            s["cut"] += 1
        else:
            s["valid" if ok else "invalid"] += 1
        if tree is None:
            tree = _parse_prefix(code)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n, n_ann, ret, full = func_hints(node)
                s["defs"] += 1
                s["full"] += full
                s["any_hint"] += (n_ann > 0 or ret)
    return s


_KW = {"if", "for", "while", "switch", "catch", "function", "return", "with", "else"}
_FUNC = re.compile(r"function\s*\*?\s*[\w$]*\s*(?:<[^>()]*>)?\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*(:\s*[^{;]+?)?\s*\{")
_ARROW = re.compile(r"(?:const|let|var)\s+[\w$]+\s*(?::[^=]+?)?=\s*(?:async\s*)?(?:<[^>()]*>)?\s*\(([^()]*)\)\s*(:\s*[^=]+?)?\s*=>")
_METHOD = re.compile(r"^[ \t]*(?:(?:public|private|protected|static|async|get|set|readonly)\s+)*([\w$]+)\s*(?:<[^>()]*>)?"
                     r"\s*\(([^()]*)\)\s*(:\s*[^{;]+?)?\s*\{", re.M)


def _split_params(p):
    out, depth, cur = [], 0, ""
    for ch in p:
        if ch in "<({[":
            depth += 1
        elif ch in ">)}]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out if x.strip()]


def _annot(param):
    """A parameter has a type if it has ':' before any default '='."""
    head = param.split("=")[0]
    return ":" in head and not head.lstrip().startswith("{")


def js_stats(reply):
    """{blocks, funcs, annotated, full} for one reply (counts)."""
    s = dict(blocks=0, funcs=0, annotated=0, full=0)
    for code, _ in blocks(reply, JS_TAGS):
        s["blocks"] += 1
        found = [(m.group(1), m.group(2), False) for m in _FUNC.finditer(code)]
        found += [(m.group(1), m.group(2), False) for m in _ARROW.finditer(code)]
        for m in _METHOD.finditer(code):
            if m.group(1) not in _KW:
                found.append((m.group(2), m.group(3), m.group(1) == "constructor"))
        for params, ret, ctor in found:
            ps = _split_params(params)
            n_ann = sum(map(_annot, ps))
            s["funcs"] += 1
            s["annotated"] += (n_ann > 0 or bool(ret))
            s["full"] += bool(n_ann == len(ps) and (ret or ctor))
    return s


C_ARROW = re.compile(r"^[ \t]*[\w*\s]+\([^()]*\)\s*->\s*[\w *]+\s*\{", re.M)


def c_arrow(reply):
    """1 if a ```c block has a Python-style return annotation after a C signature (`int f(int x) -> int {`), else 0."""
    return int(any(C_ARROW.search(code) for code, _ in blocks(reply, {"c", "cpp", "c++", "h"})))
