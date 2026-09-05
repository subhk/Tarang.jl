#!/usr/bin/env python3
"""Fold `ensure_layout!(x, :L)` immediately followed by `get_<L>_data(x)` into
`<L>_data!(x)`.

Only the adjacent, same-operand pattern is touched. `grid_data!` / `coeff_data!`
(src/core/field/field_layout/field_layout_access.jl) are defined as exactly that
pair, so the rewrite cannot change behaviour. Everything else — a layout set for
a read that happens later or elsewhere — is left for the layout ratchet
(test/test_layout_discipline_ratchet.jl) to count.

Run from the repository root:  python3 scripts/fold_layout_pairs.py
"""
import os
import re

ROOT = "src"
ENS = re.compile(r'^(\s*)ensure_layout!\(\s*([A-Za-z_][\w\.\[\]]*)\s*,\s*:(g|c)\s*\)\s*$')
ACC = {"g": ("get_grid_data", "grid_data!"), "c": ("get_coeff_data", "coeff_data!")}

folded = 0
for d, _, files in os.walk(ROOT):
    for f in files:
        if not f.endswith(".jl") or f == "field_layout_access.jl":
            continue
        p = os.path.join(d, f)
        L = open(p).read().split("\n")
        out, i, changed = [], 0, False
        while i < len(L):
            m = ENS.match(L[i])
            if m:
                x, lay = m.group(2), m.group(3)
                getter, acc = ACC[lay]
                j = i + 1
                while j < len(L) and (L[j].strip() == "" or L[j].strip().startswith("#")):
                    j += 1
                pat = re.compile(r'\b' + getter + r'\(\s*' + re.escape(x) + r'\s*\)')
                if j < len(L) and pat.search(L[j]) and "ensure_layout!" not in L[j]:
                    out.extend(L[i + 1:j])              # keep blank/comment lines
                    out.append(pat.sub(acc + "(" + x + ")", L[j], count=1))
                    i = j + 1
                    folded += 1
                    changed = True
                    continue
            out.append(L[i])
            i += 1
        if changed:
            open(p, "w").write("\n".join(out))
print("folded", folded)
