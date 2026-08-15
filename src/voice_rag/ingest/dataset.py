"""Load a working subset of ai4bharat/MSMARCO-XI.

The full dump is ~55GB / 11.5M rows. We take the Hindi validation parquet
(English + Hindi aligned) and sample a compact, gold-rich subset so ingest
and the 200ms query path stay realistic for a local demo.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from voice_rag.config import settings
from voice_rag.textutil import content_hash, normalize


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    return [value]


def _passages_from_row(row: dict) -> dict[str, list]:
    passages = row.get("passages") or {}
    if hasattr(passages, "as_py"):
        passages = passages.as_py()
    if not isinstance(passages, dict):
        return {"is_selected": [], "English_passages": [], "Translated_passages": []}
    return {
        "is_selected": [int(x) for x in _as_list(passages.get("is_selected"))],
        "English_passages": [str(x) for x in _as_list(passages.get("English_passages"))],
        "Translated_passages": [str(x) for x in _as_list(passages.get("Translated_passages"))],
    }


def load_rows(path: Path | None = None, limit: int | None = None) -> list[dict]:
    path = path or settings.parquet_path
    if not path.exists():
        raise FileNotFoundError(
            f"MSMARCO-XI parquet not found at {path}. "
            "Download validation/hinval.parquet from "
            "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI"
        )
    cols_keep = [
        "query_id",
        "query_type",
        "query",
        "Answer",
        "Eng_Query",
        "Eng_Answer",
        "passages",
    ]
    table = pq.read_table(path, columns=cols_keep)
    n_total = table.num_rows
    if limit is None or limit >= n_total:
        chosen = table
        n = n_total
    else:
        import numpy as np

        idx = np.linspace(0, n_total - 1, limit, dtype=np.int64)
        chosen = table.take(idx)
        n = chosen.num_rows
    del table
    cols = chosen.to_pydict()
    del chosen
    keys = list(cols.keys())
    rows = []
    for i in range(n):
        row = {k: cols[k][i] for k in keys}
        rows.append(row)
    return rows


def build_corpus(
    rows: list[dict],
    languages: tuple[str, ...] = ("en", "hi"),
) -> dict[str, Any]:
    """Deduplicate passages, keep gold links, and collect eval queries."""
    parents: dict[str, dict] = {}
    hash_to_pid: dict[str, str] = {}
    queries: list[dict] = []

    def add_passage(
        text: str,
        language: str,
        query_id: int,
        query_type: str,
        is_gold: bool,
    ) -> str | None:
        text = normalize(text)
        if len(text) < 40:
            return None
        h = content_hash(text)
        if h in hash_to_pid:
            pid = hash_to_pid[h]
            rec = parents[pid]
            if query_id not in rec["source_query_ids"]:
                rec["source_query_ids"].append(query_id)
            if query_type and query_type not in rec["query_types"]:
                rec["query_types"].append(query_type)
            rec["is_gold"] = rec["is_gold"] or is_gold
            return pid
        pid = f"{language}-{h}"
        hash_to_pid[h] = pid
        parents[pid] = {
            "parent_id": pid,
            "text": text,
            "language": language,
            "query_types": [query_type] if query_type else [],
            "source_query_ids": [query_id],
            "is_gold": is_gold,
        }
        return pid

    for row in rows:
        qid = int(row.get("query_id") or 0)
        qtype = str(row.get("query_type") or "UNKNOWN")
        passages = _passages_from_row(row)
        selected = passages["is_selected"]
        en_pass = passages["English_passages"]
        hi_pass = passages["Translated_passages"]
        gold_en: list[str] = []
        gold_hi: list[str] = []

        n = max(len(en_pass), len(hi_pass), len(selected))
        for i in range(n):
            is_gold = bool(selected[i]) if i < len(selected) else False
            if "en" in languages and i < len(en_pass):
                pid = add_passage(en_pass[i], "en", qid, qtype, is_gold)
                if is_gold and pid:
                    gold_en.append(pid)
            if "hi" in languages and i < len(hi_pass):
                pid = add_passage(hi_pass[i], "hi", qid, qtype, is_gold)
                if is_gold and pid:
                    gold_hi.append(pid)

        queries.append(
            {
                "query_id": qid,
                "query_type": qtype,
                "en_query": normalize(str(row.get("Eng_Query") or "")),
                "hi_query": normalize(str(row.get("query") or "")),
                "en_answer": normalize(str(row.get("Eng_Answer") or "")),
                "hi_answer": normalize(str(row.get("Answer") or "")),
                "gold_en": gold_en,
                "gold_hi": gold_hi,
            }
        )

    return {
        "parents": list(parents.values()),
        "queries": queries,
        "stats": {
            "rows": len(rows),
            "unique_passages": len(parents),
            "by_language": _count_lang(parents),
            "gold_passages": sum(1 for p in parents.values() if p["is_gold"]),
        },
    }


def _count_lang(parents: dict[str, dict]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for p in parents.values():
        c[p["language"]] += 1
    return dict(c)


def split_queries(queries: list[dict], holdout: int) -> tuple[list[dict], list[dict]]:
    usable = [q for q in queries if q["en_query"] and q["gold_en"]]
    holdout = min(holdout, max(1, len(usable) // 5))
    # take every k-th query so types stay mixed
    step = max(1, len(usable) // holdout)
    held = usable[::step][:holdout]
    held_ids = {q["query_id"] for q in held}
    train = [q for q in queries if q["query_id"] not in held_ids]
    return train, held


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
