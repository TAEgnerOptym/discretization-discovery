import re
import math
import pandas as pd

URLS = [
    "https://hjemmesider.diku.dk/~spooren/solomon/c1c2solu.htm",
    "https://hjemmesider.diku.dk/~spooren/solomon/r1r2solu.htm",
    "https://hjemmesider.diku.dk/~spooren/solomon/rc12solu.htm",
]

MISSING = 999_999_999

# Patterns for instance ids (C101, C201, R101, R201, RC101, RC201, …)
INST_RE = re.compile(r"^(?:RC|C|R)[12]\d{2}$", re.IGNORECASE)

# columns that might contain the objective value (priority order)
OBJ_CANDIDATE_NAMES = [
    "best", "best solution", "best sol", "best known", "obj", "objective", "distance", "dist"
]

# possible problem-size columns if the table contains multiple (e.g., 25/50/100)
SIZE_NAMES = ["25", "50", "100", 25, 50, 100, "n=25", "n=50", "n=100"]

def _to_num(x):
    """Convert likely numeric to float; return None for blanks/NaN."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    s = str(x).strip()
    if not s or s in {"-", "—"}:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

def _find_instance_col(df: pd.DataFrame):
    """Find the column that contains instance IDs like C101, RC201, …"""
    for col in df.columns:
        hits = 0
        for v in df[col]:
            if isinstance(v, str) and INST_RE.match(v.strip()):
                hits += 1
        if hits >= max(3, len(df) // 10):  # heuristic
            return col
    return None

def _find_obj_col(df: pd.DataFrame):
    """Pick an objective column by name, else the first mostly-numeric column."""
    lower_map = {c: str(c).lower() for c in df.columns}
    # try by name priority
    for want in OBJ_CANDIDATE_NAMES:
        for c in df.columns:
            if want in lower_map[c]:
                return c
    # fallback: the first column with many numeric-like entries
    best_col, best_count = None, -1
    for c in df.columns:
        nums = sum(_to_num(v) is not None for v in df[c])
        if nums > best_count:
            best_col, best_count = c, nums
    return best_col

def _possible_size_cols(df: pd.DataFrame):
    """Return any columns that look like separate problem-size objectives."""
    cols = []
    lowers = {c: str(c).lower() for c in df.columns}
    for c in df.columns:
        txt = lowers[c]
        for s in ("25", "50", "100"):
            if s == txt or f"{s}" in txt:
                cols.append((c, int(s)))
                break
    # de-duplicate keeping order
    seen = set()
    out = []
    for c, s in cols:
        if (c, s) not in seen:
            seen.add((c, s))
            out.append((c, s))
    return out

def fetch_solomon_objectives():
    """
    Returns: dict with keys (dataset, problem_size) and float objective.
             Missing values -> 999_999_999
    """
    result = {}
    for url in URLS:
        tables = pd.read_html(url, flavor="bs4")
        for df in tables:
            # normalize headers
            df.columns = [str(c).strip() for c in df.columns]
            inst_col = _find_instance_col(df)
            if inst_col is None:
                continue

            # case A: table has separate columns per size (25/50/100)
            size_cols = _possible_size_cols(df)
            if size_cols:
                for _, row in df.iterrows():
                    inst = row.get(inst_col)
                    if not isinstance(inst, str) or not INST_RE.match(inst.strip()):
                        continue
                    inst = inst.strip().upper()
                    for col, sz in size_cols:
                        val = _to_num(row.get(col))
                        result[(inst, sz)] = val if val is not None else MISSING
                continue

            # case B: single objective column (assume size=100 by default)
            obj_col = _find_obj_col(df)
            if obj_col is None:
                continue
            for _, row in df.iterrows():
                inst = row.get(inst_col)
                if not isinstance(inst, str) or not INST_RE.match(inst.strip()):
                    continue
                inst = inst.strip().upper()
                val = _to_num(row.get(obj_col))
                # fall back to 100 as Solomon’s standard instance size
                result[(inst, 100)] = val if val is not None else MISSING

    return result

if __name__ == "__main__":
    solomon_obj = fetch_solomon_objectives()
    # example lookup:
    # print(solomon_obj.get(("C101", 100)))
    print(f"Loaded {len(solomon_obj)} entries")
