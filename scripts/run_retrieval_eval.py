import json
import time
from collections import defaultdict
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def matches_expected_any_of(answer: str, expected_any_of: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(candidate.lower() in answer_lower for candidate in expected_any_of)


def normalize_text(text: str) -> str:
    return text.lower().strip()


def is_refusal_answer(text: str) -> bool:
    normalized = normalize_text(text)
    refusal_markers = [
        "无法回答",
        "无法确定",
        "无法从提供的上下文中确定",
        "根据当前检索内容无法确定",
        "cannot be determined",
        "cannot determine",
        "not enough information",
        "does not provide information",
        "does not contain information",
        "not provided in the context",
        "the context does not provide",
    ]
    return any(marker in normalized for marker in refusal_markers)


def case_passes(case: dict, answer: str, status_code: int) -> bool:
    if status_code != 200:
        return False

    expected_any_of = case["expected_any_of"]
    if matches_expected_any_of(answer, expected_any_of):
        return True

    if case.get("category") in {"no_answer", "confusion"}:
        return is_refusal_answer(answer)

    return False


def classify_failure(result: dict) -> str | None:
    if result["passed"]:
        return None

    if result["status_code"] != 200:
        error_text = str(result.get("error", "")).lower()

        infra_markers = [
            "429",
            "too many requests",
            "ssl",
            "httpsconnectionpool",
            "max retries exceeded",
            "connection error",
            "connection reset",
            "timeout",
            "timed out",
            "eof",
        ]
        if any(marker in error_text for marker in infra_markers):
            return "infra_error"

        return "request_error"

    return "quality_failure"


def summarize_failures(results: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for r in results:
        failure_type = r.get("failure_type")
        if failure_type is not None:
            counts[failure_type] += 1
    return dict(counts)


# Retry helper for transient infra failures.
def post_with_retry(client: TestClient, payload: dict, max_attempts: int = 3) -> tuple:
    last_resp = None
    last_data = None

    for attempt in range(1, max_attempts + 1):
        resp = client.post("/ask_debug", json=payload)
        data = resp.json()

        if resp.status_code == 200:
            return resp, data

        error_text = str(data).lower()
        retry_markers = [
            "429",
            "too many requests",
            "ssl",
            "httpsconnectionpool",
            "max retries exceeded",
            "connection error",
            "connection reset",
            "timeout",
            "timed out",
            "eof",
        ]

        last_resp = resp
        last_data = data

        should_retry = any(marker in error_text for marker in retry_markers)
        if not should_retry or attempt == max_attempts:
            return resp, data

        time.sleep(1.5 * attempt)

    return last_resp, last_data


def run_case(client: TestClient, case: dict) -> list[dict]:
    results = []

    for mode in case["retrieval_modes"]:
        for rerank_mode in case.get("rerank_modes", ["none"]):
            payload = {
                "question": case["question"],
                "file_id": case["file_id"],
                "top_k": case.get("top_k", 3),
                "retrieve_top_k": case.get("retrieve_top_k", 10),
                "retrieval_mode": mode,
                "rerank_mode": rerank_mode,
            }

            resp, data = post_with_retry(client, payload)

            answer = data.get("answer", "")
            citations = data.get("citations", [])
            hits = data.get("hits", [])

            expected_any_of = case["expected_any_of"]
            passed = case_passes(case, answer, resp.status_code)

            result = {
                "case": case["name"],
                "category": case["category"],
                "mode": mode,
                "rerank_mode": rerank_mode,
                "top_k": case.get("top_k", 3),
                "retrieve_top_k": case.get("retrieve_top_k", 10),
                "status_code": resp.status_code,
                "passed": passed,
                "question": case["question"],
                "expected_any_of": expected_any_of,
                "answer": answer,
                "citations_count": len(citations),
                "hits_count": len(hits),
            }

            if resp.status_code != 200:
                result["error"] = data
            result["failure_type"] = classify_failure(result)

            results.append(result)

    return results


def print_detailed_results(results: list[dict]) -> None:
    print("\n=== Retrieval Eval Results ===")
    for r in results:
        print(
            f"{r['case']} | {r['category']} | {r['mode']} | "
            f"rerank={r['rerank_mode']} | "
            f"top_k={r['top_k']} | retrieve_top_k={r['retrieve_top_k']} | "
            f"status={r['status_code']} | "
            f"passed={r['passed']} | citations={r['citations_count']} | "
            f"hits={r['hits_count']}"
        )


def print_summary(results: list[dict]) -> None:
    by_mode_and_rerank = defaultdict(list)
    by_category = defaultdict(list)

    for r in results:
        by_mode_and_rerank[(r["mode"], r["rerank_mode"])].append(r)
        by_category[r["category"]].append(r)

    total = len(results)
    passed_total = sum(1 for r in results if r["passed"])

    print("\n=== Summary ===")
    print(f"overall: {passed_total}/{total} passed")

    print("\n=== By Retrieval Mode + Rerank ===")
    for (mode, rerank_mode), items in by_mode_and_rerank.items():
        group_total = len(items)
        group_passed = sum(1 for r in items if r["passed"])
        avg_citations = sum(r["citations_count"] for r in items) / group_total
        avg_hits = sum(r["hits_count"] for r in items) / group_total

        print(
            f"{mode} + {rerank_mode}: {group_passed}/{group_total} passed | "
            f"avg_citations={avg_citations:.2f} | avg_hits={avg_hits:.2f}"
        )

    print("\n=== By Category ===")
    for category, items in by_category.items():
        category_total = len(items)
        category_passed = sum(1 for r in items if r["passed"])
        print(f"{category}: {category_passed}/{category_total} passed")

    failure_summary = summarize_failures(results)

    print("\n=== Failure Types ===")
    if not failure_summary:
        print("None")
    else:
        for failure_type, count in failure_summary.items():
            print(f"{failure_type}: {count}")


def print_failures(results: list[dict]) -> None:
    failures = [r for r in results if not r["passed"]]

    print("\n=== Failures ===")
    if not failures:
        print("None")
        return

    for r in failures:
        print(
            f"- case={r['case']} category={r['category']} "
            f"mode={r['mode']} rerank={r['rerank_mode']} "
            f"failure_type={r.get('failure_type')}"
        )
        print(f"  question: {r['question']}")
        print(f"  expected_any_of: {r['expected_any_of']}")
        print(f"  answer: {r['answer']}")
        if "error" in r:
            print(f"  error: {r['error']}")


def save_results(
    results: list[dict],
    output_path: str = "evals/retrieval_eval_results.json",
) -> None:
    Path(output_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved results to {output_path}")


def build_summary(results: list[dict]) -> dict:
    by_mode_and_rerank = defaultdict(list)
    by_category = defaultdict(list)

    for r in results:
        by_mode_and_rerank[(r["mode"], r["rerank_mode"])].append(r)
        by_category[r["category"]].append(r)

    total = len(results)
    passed_total = sum(1 for r in results if r["passed"])

    mode_summary = {}
    for (mode, rerank_mode), items in by_mode_and_rerank.items():
        group_total = len(items)
        group_passed = sum(1 for r in items if r["passed"])
        mode_summary[f"{mode} + {rerank_mode}"] = {
            "passed": group_passed,
            "total": group_total,
            "avg_citations": round(
                sum(r["citations_count"] for r in items) / group_total, 2
            ),
            "avg_hits": round(
                sum(r["hits_count"] for r in items) / group_total, 2
            ),
        }

    category_summary = {}
    for category, items in by_category.items():
        category_total = len(items)
        category_passed = sum(1 for r in items if r["passed"])
        category_summary[category] = {
            "passed": category_passed,
            "total": category_total,
        }

    failures = [r for r in results if not r["passed"]]
    failure_summary = summarize_failures(results)

    return {
        "overall": {
            "passed": passed_total,
            "total": total,
        },
        "by_mode": mode_summary,
        "by_category": category_summary,
        "failures": failures,
        "failure_summary": failure_summary,
    }


def save_markdown_report(
    results: list[dict],
    output_path: str = "evals/retrieval_eval_report.md",
) -> None:
    summary = build_summary(results)

    lines: list[str] = []
    lines.append("# Retrieval Evaluation Report")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        f"- Passed: **{summary['overall']['passed']}/{summary['overall']['total']}**"
    )
    lines.append("")

    lines.append("## By Retrieval Mode + Rerank")
    lines.append("")
    lines.append("| Mode | Passed | Avg Citations | Avg Hits |")
    lines.append("|---|---:|---:|---:|")
    for mode, stats in summary["by_mode"].items():
        lines.append(
            f"| {mode} | {stats['passed']}/{stats['total']} | "
            f"{stats['avg_citations']:.2f} | {stats['avg_hits']:.2f} |"
        )
    lines.append("")

    lines.append("## By Category")
    lines.append("")
    lines.append("| Category | Passed |")
    lines.append("|---|---:|")
    for category, stats in summary["by_category"].items():
        lines.append(
            f"| {category} | {stats['passed']}/{stats['total']} |"
        )
    lines.append("")

    lines.append("## Failure Types")
    lines.append("")
    if not summary["failure_summary"]:
        lines.append("None")
    else:
        lines.append("| Failure Type | Count |")
        lines.append("|---|---:|")
        for failure_type, count in summary["failure_summary"].items():
            lines.append(f"| {failure_type} | {count} |")
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")
    lines.append(
        "| Case | Category | Mode | Rerank | top_k | " \
        "retrieve_top_k | Status | Passed | Citations | Hits |"
    )
    lines.append(
        "|---|---|---|---|---:|---:|---:|---|---:|---:|"
    )
    for r in results:
        lines.append(
            f"| {r['case']} | {r['category']} | {r['mode']} | {r['rerank_mode']} | "
            f"{r['top_k']} | {r['retrieve_top_k']} | {r['status_code']} | "
            f"{r['passed']} | {r['citations_count']} | {r['hits_count']} |"
        )
    lines.append("")

    lines.append("## Failures")
    lines.append("")
    if not summary["failures"]:
        lines.append("None")
        lines.append("")
    else:
        for r in summary["failures"]:
            lines.append(
                f"### {r['case']} ({r['category']} / {r['mode']} / {r['rerank_mode']})"
            )
            lines.append("")
            lines.append(f"- Question: `{r['question']}`")
            lines.append(f"- Expected any of: `{r['expected_any_of']}`")
            lines.append(f"- Answer: `{r['answer']}`")
            lines.append(f"- Failure type: `{r.get('failure_type')}`")
            if "error" in r:
                lines.append(f"- Error: `{r['error']}`")
            lines.append("")

    lines.append("## Key Takeaways")
    lines.append("")
    lines.append(
        "- This report compares `bm25`, `vector`, and `hybrid` retrieval modes on local eval cases."
    )
    lines.append(
        "- It is intended to support retrieval debugging, tuning decisions, "
        "and regression checking."
    )
    lines.append(
        "- Refusal-style cases are graded with a normalized refusal matcher "
        "so semantically correct no-answer responses are not penalized for "
        "wording differences."
    )
    lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved markdown report to {output_path}")


def main():
    cases = json.loads(
        Path("evals/retrieval_eval_cases.json").read_text(encoding="utf-8")
    )

    all_results = []

    with TestClient(app) as client:
        for case in cases:
            all_results.extend(run_case(client, case))

    print_detailed_results(all_results)
    print_summary(all_results)
    print_failures(all_results)
    save_results(all_results)
    save_markdown_report(all_results)


if __name__ == "__main__":
    main()