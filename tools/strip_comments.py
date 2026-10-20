import ast
import io
import sys
import tokenize
from pathlib import Path


def strip(src):
    tree = ast.parse(src)
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                drop.update(range(first.lineno, first.end_lineno + 1))
    lines = src.splitlines(keepends=True)
    cuts = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            cuts[tok.start[0]] = tok.start[1]
    out = []
    for n, line in enumerate(lines, 1):
        if n in drop:
            continue
        if n in cuts:
            kept = line[: cuts[n]].rstrip()
            if not kept:
                continue
            line = kept + "\n"
        out.append(line)
    text = "".join(out)
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text.lstrip("\n")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        p = Path(p)
        s = p.read_text(encoding="utf-8")
        t = strip(s)
        ast.parse(t)
        if t != s:
            p.write_text(t, encoding="utf-8")
            print("stripped", p)
