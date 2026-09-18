from searching_strategy.hybrid_search import hybrid_search
from generate import answer_question, build_context, llm


TOP_K = 3

# Fixed test set for the currently-ingested project (ExpenseTracker).
# "expected_files" are substrings matched against a retrieved chunk's
# absolute file path, so this stays portable across machines/checkouts.
EVAL_CASES = [
    {
        "question": "Where is register_user implemented?",
        "expected_files": ["Backend/app.py"],
    },
    {
        "question": "What does the login function do?",
        "expected_files": ["Backend/app.py"],
    },
    {
        "question": "What does the User model look like?",
        "expected_files": ["Backend/models.py"],
    },
    {
        "question": "What are all the libraries we imported in backend files?",
        "expected_files": ["Backend/app.py", "Backend/models.py"],
    },
    {
        "question": "How does the frontend show a list of expenses?",
        "expected_files": ["ViewExpenses.jsx"],
    },
]


# ==================================================
# RETRIEVAL METRICS
# ==================================================

def is_relevant(file_path, expected_files):

    return any(
        expected in file_path
        for expected in expected_files
    )


def retrieval_metrics(question, expected_files, top_k=TOP_K):

    results = hybrid_search(question, top_k=top_k)

    retrieved_files = [
        result.metadata["file"] for result in results
    ]

    relevant_flags = [
        is_relevant(file_path, expected_files)
        for file_path in retrieved_files
    ]

    hit = 1.0 if any(relevant_flags) else 0.0

    precision = (
        sum(relevant_flags) / len(relevant_flags)
        if relevant_flags else 0.0
    )

    found_expected = {
        expected
        for expected in expected_files
        for file_path in retrieved_files
        if expected in file_path
    }

    recall = (
        len(found_expected) / len(expected_files)
        if expected_files else 0.0
    )

    mrr = 0.0

    for rank, relevant in enumerate(relevant_flags, start=1):

        if relevant:
            mrr = 1.0 / rank
            break

    return {
        "hit": hit,
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "retrieved_files": retrieved_files,
        "results": results,
    }


# ==================================================
# FAITHFULNESS (LLM-as-judge)
# ==================================================

FAITHFULNESS_PROMPT = """
You are grading whether an answer is faithful to the given context.

Context:
{context}

Answer:
{answer}

Is every claim in the answer supported by the context? Reply with
exactly one word on the first line: YES or NO. Then on a new line,
briefly say which claims (if any) are not supported.
"""


def faithfulness_check(context, answer):

    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        answer=answer,
    )

    response = llm.invoke([
        {
            "role": "user",
            "content": prompt,
        }
    ])

    verdict = response.content.strip()

    is_faithful = verdict.upper().startswith("YES")

    return is_faithful, verdict


# ==================================================
# RUN
# ==================================================

def run_eval():

    retrieval_results = []
    faithfulness_flags = []

    for case in EVAL_CASES:

        question = case["question"]
        expected_files = case["expected_files"]

        metrics = retrieval_metrics(question, expected_files)
        retrieval_results.append(metrics)

        context = build_context(metrics["results"])
        answer = answer_question([], question, top_k=TOP_K)

        is_faithful, verdict = faithfulness_check(context, answer)
        faithfulness_flags.append(is_faithful)

        print(f"Q: {question}")

        print(
            f"  hit={metrics['hit']:.0f} "
            f"precision={metrics['precision']:.2f} "
            f"recall={metrics['recall']:.2f} "
            f"mrr={metrics['mrr']:.2f} "
            f"faithful={is_faithful}"
        )

        if not metrics["hit"]:
            print(f"  expected: {expected_files}")
            print(f"  retrieved: {metrics['retrieved_files']}")

        if not is_faithful:
            print(f"  faithfulness verdict: {verdict}")

        print()

    n = len(EVAL_CASES)

    print("========== SUMMARY ==========")
    print(f"Cases:          {n}")
    print(f"Hit Rate@{TOP_K}:    {sum(r['hit'] for r in retrieval_results) / n:.2f}")
    print(f"Precision@{TOP_K}:   {sum(r['precision'] for r in retrieval_results) / n:.2f}")
    print(f"Recall@{TOP_K}:      {sum(r['recall'] for r in retrieval_results) / n:.2f}")
    print(f"MRR:            {sum(r['mrr'] for r in retrieval_results) / n:.2f}")
    print(f"Faithfulness:   {sum(faithfulness_flags) / n:.2f}")


if __name__ == "__main__":
    run_eval()
