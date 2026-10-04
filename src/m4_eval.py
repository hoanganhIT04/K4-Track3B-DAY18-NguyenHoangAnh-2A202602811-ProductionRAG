from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not questions:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    try:
        import pandas as pd
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df = result.to_pandas()

        per_question = []
        for _, row in df.iterrows():
            f_val = float(row["faithfulness"]) if "faithfulness" in row and not pd.isna(row["faithfulness"]) else 0.0
            ar_val = float(row["answer_relevancy"]) if "answer_relevancy" in row and not pd.isna(row["answer_relevancy"]) else 0.0
            cp_val = float(row["context_precision"]) if "context_precision" in row and not pd.isna(row["context_precision"]) else 0.0
            cr_val = float(row["context_recall"]) if "context_recall" in row and not pd.isna(row["context_recall"]) else 0.0

            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=f_val,
                answer_relevancy=ar_val,
                context_precision=cp_val,
                context_recall=cr_val,
            ))

        avg_f = float(df["faithfulness"].mean()) if "faithfulness" in df and not df["faithfulness"].isnull().all() else 0.0
        avg_ar = float(df["answer_relevancy"].mean()) if "answer_relevancy" in df and not df["answer_relevancy"].isnull().all() else 0.0
        avg_cp = float(df["context_precision"].mean()) if "context_precision" in df and not df["context_precision"].isnull().all() else 0.0
        avg_cr = float(df["context_recall"].mean()) if "context_recall" in df and not df["context_recall"].isnull().all() else 0.0

        return {
            "faithfulness": round(avg_f, 4),
            "answer_relevancy": round(avg_ar, 4),
            "context_precision": round(avg_cp, 4),
            "context_recall": round(avg_cr, 4),
            "per_question": per_question
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        per_question = []
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            per_question.append(EvalResult(
                question=q, answer=a, contexts=c, ground_truth=gt,
                faithfulness=0.0, answer_relevancy=0.0,
                context_precision=0.0, context_recall=0.0
            ))
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": per_question}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa câu trả lời ngoài tài liệu (Hallucination)",
            "Thắt chặt system prompt, giảm temperature về 0"
        ),
        "context_recall": (
            "Hệ thống tìm kiếm bỏ sót đoạn văn đúng (Missing relevant chunks)",
            "Cải thiện chunking hoặc bổ sung từ khóa BM25"
        ),
        "context_precision": (
            "Đoạn văn không liên quan bị xếp lên đầu (Noise in context)",
            "Bổ sung Cross-Encoder reranking hoặc lọc theo metadata"
        ),
        "answer_relevancy": (
            "Câu trả lời bị lệch trọng tâm câu hỏi (Irrelevant answer)",
            "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn"
        ),
    }

    analyzed = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg_score = sum(metrics.values()) / 4.0
        worst_metric = min(metrics.keys(), key=lambda k: metrics[k])
        diag, fix = diagnostic_tree.get(
            worst_metric,
            ("LLM or Retrieval issue", "Review pipeline step")
        )

        analyzed.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "avg_score": round(avg_score, 4),
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "diagnosis": diag,
            "suggested_fix": fix,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
