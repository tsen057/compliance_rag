"""
scripts/evaluate.py
───────────────────
RAGAS evaluation of the RAG pipeline.

Metrics:
  - faithfulness        : Are answers grounded in the retrieved context?
  - answer_relevancy    : Does the answer address the question?
  - context_precision   : Are retrieved chunks relevant to the question?

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --output results/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)

# ── Evaluation dataset ─────────────────────────────────────────────────────────
# These Q&A pairs are ground-truth examples for Basel III / FATF content.
# Extend this list as you add more documents.

EVAL_DATASET = [
    {
        "question": "What is the minimum Common Equity Tier 1 ratio under Basel III?",
        "ground_truth": "Under Basel III, banks must maintain a minimum Common Equity Tier 1 (CET1) ratio of 4.5% of risk-weighted assets.",
    },
    {
        "question": "What is the capital conservation buffer introduced by Basel III?",
        "ground_truth": "Basel III introduced a capital conservation buffer of 2.5%, made up of Common Equity Tier 1 capital, bringing the effective minimum CET1 requirement to 7%.",
    },
    {
        "question": "What is the leverage ratio requirement under Basel III?",
        "ground_truth": "Basel III introduced a non-risk-based leverage ratio of 3%, calculated as Tier 1 capital divided by total exposure.",
    },
    {
        "question": "What is customer due diligence under FATF recommendations?",
        "ground_truth": "Customer due diligence (CDD) under FATF recommendations requires financial institutions to identify and verify customer identity, understand the nature of the business relationship, and conduct ongoing monitoring of transactions.",
    },
    {
        "question": "What is the Liquidity Coverage Ratio?",
        "ground_truth": "The Liquidity Coverage Ratio (LCR) requires banks to hold a sufficient stock of unencumbered high-quality liquid assets to cover total net cash outflows over a 30-day stress period, with a minimum ratio of 100%.",
    },
]


def run_evaluation(output_path: Path | None = None) -> dict:
    """Run RAGAS evaluation and return metrics."""
    from datasets import Dataset

    # Import agent here to avoid loading models at module level
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.core.agent import ComplianceAgent

    logger.info("Loading ComplianceAgent...")
    agent = ComplianceAgent()

    logger.info(f"Running evaluation on {len(EVAL_DATASET)} questions...")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(EVAL_DATASET, 1):
        logger.info(f"  [{i}/{len(EVAL_DATASET)}] {item['question'][:60]}...")
        result = agent.query(item["question"])

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([s["excerpt"] for s in result["sources"]])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    logger.info("Computing RAGAS metrics...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    results = {
        "faithfulness": round(scores["faithfulness"], 4),
        "answer_relevancy": round(scores["answer_relevancy"], 4),
        "context_precision": round(scores["context_precision"], 4),
        "num_questions": len(EVAL_DATASET),
    }

    logger.info("\n=== RAGAS Evaluation Results ===")
    logger.info(f"  Faithfulness      : {results['faithfulness']:.2%}")
    logger.info(f"  Answer Relevancy  : {results['answer_relevancy']:.2%}")
    logger.info(f"  Context Precision : {results['context_precision']:.2%}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for Compliance RAG")
    parser.add_argument("--output", type=Path, default=None, help="Path to save JSON results")
    args = parser.parse_args()

    run_evaluation(output_path=args.output)
