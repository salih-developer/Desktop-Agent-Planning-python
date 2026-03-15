"""
Model benchmark: SQL + C# code writing quality test.
Usage: python benchmark.py
"""
from __future__ import annotations
import sys
import threading
import time
from pathlib import Path

# Windows console UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from tools.registry import build_default_registry
from core.react_loop import ReactLoop

MODELS = [
    "qwen3.5:35b",
    "gpt-oss:20b",
    "qwen3-coder-32k:latest",
]

TASKS = [
    {
        "id": "sql_1",
        "label": "SQL StoredProc",
        "query": (
            "Asagidaki tablolari kullanan bir SQL Server stored procedure yaz. "
            "Procedure adi: sp_GetCustomerOrders. "
            "Tablolar: Customers(CustomerID, Name, Email), "
            "Orders(OrderID, CustomerID, OrderDate, TotalAmount), "
            "OrderItems(ItemID, OrderID, ProductName, Quantity, UnitPrice). "
            "Parametre: @CustomerID int. "
            "Donursun: musteri adi, email, tum siparisler (tarih, toplam tutar), "
            "her siparisin kalemleri (urun adi, adet, birim fiyat). "
            "Performans icin JOIN, nolock ve NULL kontrolu ekle."
        ),
    },
    {
        "id": "cs_1",
        "label": "C# Repository",
        "query": (
            "C# ile generic Repository pattern yaz. "
            "IRepository<T> interface ve EntityFramework Core kullanan "
            "Repository<T> implementasyonu olsun. "
            "Metodlar: GetByIdAsync, GetAllAsync, FindAsync(expression), "
            "AddAsync, Update, Delete, SaveChangesAsync. "
            "Async/await kullan, DbContext injection ile calissin. "
            "Ayrica IUnitOfWork ve UnitOfWork implementasyonunu da ekle."
        ),
    },
]


def run_task(model: str, task: dict, cfg) -> dict:
    cancel = threading.Event()
    registry = build_default_registry()
    for tool in registry.all_tools():
        if hasattr(tool, "workspace"):
            tool.workspace = cfg.workspace

    loop = ReactLoop(
        base_url=cfg.ollama_base_url,
        model=model,
        registry=registry,
        cancel_event=cancel,
        max_iterations=10,
        traces_dir=Path(cfg.traces_dir),
    )

    tool_fails = 0
    tool_oks = 0

    def on_out(chunk: str):
        nonlocal tool_fails, tool_oks
        if "STATUS: FAIL" in chunk:
            tool_fails += 1
        if "STATUS: OK" in chunk:
            tool_oks += 1

    t0 = time.time()
    answer, executions = loop.run(
        user_query=task["query"],
        conversation_history=[],
        on_output_chunk=on_out,
    )
    elapsed = round(time.time() - t0, 1)

    return {
        "model":      model,
        "task_id":    task["id"],
        "task_label": task["label"],
        "tool_ok":    tool_oks,
        "tool_fail":  tool_fails,
        "iterations": len(executions),
        "answer_len": len(answer),
        "elapsed":    elapsed,
        "answer":     answer,
    }


def score(r: dict) -> float:
    s = 100.0
    s -= r["tool_fail"] * 15
    s -= max(0, r["iterations"] - 3) * 5
    if r["answer_len"] < 400:
        s -= 25
    return round(max(0.0, s), 1)


def print_sep(char="=", n=70):
    print(char * n)


def main():
    cfg = load_config()

    all_results: list[dict] = []

    for model in MODELS:
        print_sep()
        print(f"MODEL: {model}")
        print_sep()
        for task in TASKS:
            print(f"\n  [{task['id']}] {task['label']} ...", flush=True)
            try:
                r = run_task(model, task, cfg)
                r["score"] = score(r)
                all_results.append(r)
                print(f"  OK  tool_ok={r['tool_ok']}  tool_fail={r['tool_fail']}  "
                      f"iter={r['iterations']}  {r['elapsed']}s  "
                      f"ans={r['answer_len']} chars  PUAN={r['score']}")
            except Exception as e:
                print(f"  HATA: {e}")
                all_results.append({
                    "model": model, "task_id": task["id"],
                    "task_label": task["label"],
                    "tool_ok": 0, "tool_fail": 0,
                    "iterations": 0, "answer_len": 0,
                    "elapsed": 0, "answer": f"ERROR: {e}", "score": 0,
                })

    # Sonuc tablosu
    print("\n")
    print_sep()
    print("SONUCLAR")
    print_sep()
    fmt = "{:<36} {:<14} {:>5} {:>5} {:>7} {:>6} {:>6}"
    print(fmt.format("Model", "Gorev", "FAIL", "Iter", "Sure(s)", "Cevap", "PUAN"))
    print_sep("-")
    for r in all_results:
        print(fmt.format(
            r["model"][:35], r["task_id"],
            r["tool_fail"], r["iterations"],
            r["elapsed"], r["answer_len"], r["score"],
        ))

    # Toplam siralama
    print("\n")
    print_sep()
    print("MODEL SIRASI")
    print_sep("-")
    totals: dict[str, list[float]] = {}
    for r in all_results:
        totals.setdefault(r["model"], []).append(r["score"])
    ranked = sorted(totals.items(), key=lambda x: sum(x[1]), reverse=True)
    for i, (m, scores) in enumerate(ranked, 1):
        total = sum(scores)
        avg = total / len(scores)
        print(f"  #{i}  {m:<38}  toplam={total:.0f}  ort={avg:.1f}")

    # Cevap onizleme
    print("\n")
    print_sep()
    print("CEVAP ONIZLEME (ilk 800 char)")
    for r in all_results:
        print(f"\n--- {r['model']} / {r['task_id']} ({r['answer_len']} char) ---")
        preview = r["answer"][:800]
        print(preview.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
