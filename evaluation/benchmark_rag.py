"""
Benchmark and Evaluation Suite for BhuvanGeo-RAG.
Tests 10+ satellite-related technical questions, plus unsupported anti-hallucination probes.
Measures retrieval and answering latencies, grounding confidence, and generates a structured report.
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.config import BASE_DIR
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Test questions suite
BENCHMARK_TEST_SUITE = [
    # Supported Technical Questions (Verified Grounding Expected)
    {
        "id": "Q1_LISS4_SPECS",
        "category": "Supported - Payload Specification",
        "question": "What is the spatial resolution and swath width of LISS-IV on Resourcesat-2?",
        "expected_terms": ["5.8", "70", "LISS"],
        "expect_grounded": True,
    },
    {
        "id": "Q2_LISS3_BANDS",
        "category": "Supported - Spectral Bands",
        "question": "What spectral bands are supported by LISS-III and what is its spatial resolution?",
        "expected_terms": ["23.5", "B2", "B3", "B4", "B5", "SWIR"],
        "expect_grounded": True,
    },
    {
        "id": "Q3_AWIFS_CHARACTERISTICS",
        "category": "Supported - Payload Specification",
        "question": "What are the characteristics and swath width of the AWiFS sensor on Resourcesat-2?",
        "expected_terms": ["56", "740", "AWiFS", "camera"],
        "expect_grounded": True,
    },
    {
        "id": "Q4_CARTOSAT1_STEREO",
        "category": "Supported - Cartography & DEM",
        "question": "How does Cartosat-1 achieve along-track stereoscopic imaging for CartoDEM?",
        "expected_terms": ["Fore", "Aft", "2.5", "stereo", "DEM"],
        "expect_grounded": True,
    },
    {
        "id": "Q5_CARTOSAT2_SUBMETER",
        "category": "Supported - High Resolution Imaging",
        "question": "What is the spatial resolution and swath of Cartosat-2 series panchromatic camera?",
        "expected_terms": ["0.8", "sub-meter", "swath", "Cartosat-2"],
        "expect_grounded": True,
    },
    {
        "id": "Q6_OCM_OCEAN_COLOUR",
        "category": "Supported - Oceanography",
        "question": "What are the primary applications and spectral bands of Ocean Colour Monitor (OCM-2)?",
        "expected_terms": ["chlorophyll", "ocean", "phytoplankton", "bands"],
        "expect_grounded": True,
    },
    {
        "id": "Q7_OSCAT_SCATTEROMETER",
        "category": "Supported - Radar Meteorology",
        "question": "What type of radar payload is OSCAT on Oceansat-2 and what does it measure?",
        "expected_terms": ["scatterometer", "wind", "Ku-band"],
        "expect_grounded": True,
    },
    {
        "id": "Q8_CADASTRAL_SUITABILITY",
        "category": "Supported - Geospatial Application",
        "question": "Which sensor is best suited for village-level cadastral boundary mapping?",
        "expected_terms": ["LISS-IV", "5.8", "cadastral", "high resolution"],
        "expect_grounded": True,
    },
    {
        "id": "Q9_AWIFS_REVISIT",
        "category": "Supported - Temporal Resolution",
        "question": "What is the revisit period of the AWiFS camera across India?",
        "expected_terms": ["5 days", "5-day", "5"],
        "expect_grounded": True,
    },
    {
        "id": "Q10_RADIOMETRIC_BITS",
        "category": "Supported - Radiometry",
        "question": "What is the radiometric resolution (bit depth) of AWiFS compared to LISS-III on Resourcesat-2?",
        "expected_terms": ["12", "10", "bit"],
        "expect_grounded": True,
    },
    # Unsupported / Out-of-Domain Anti-Hallucination Probes (Strict Insufficiency Expected)
    {
        "id": "Q11_CHANDRAYAAN_ROVER",
        "category": "Anti-Hallucination Probe",
        "question": "What is the mass of the Pragyan rover on Chandrayaan-3 and its battery capacity?",
        "expected_terms": ["insufficient evidence", "insufficient"],
        "expect_grounded": False,
    },
    {
        "id": "Q12_JWST_MIRROR",
        "category": "Anti-Hallucination Probe",
        "question": "What are the beryllium primary mirror dimensions and infrared instruments of NASA's James Webb Space Telescope?",
        "expected_terms": ["insufficient evidence", "insufficient"],
        "expect_grounded": False,
    },
    {
        "id": "Q13_MARS_ROVER_SPEED",
        "category": "Anti-Hallucination Probe",
        "question": "What is the top driving speed of NASA Curiosity rover in Gale Crater on Mars?",
        "expected_terms": ["insufficient evidence", "insufficient"],
        "expect_grounded": False,
    },
]


def run_benchmark():
    """Execute evaluation suite, measure latencies, and output performance reports."""
    logger.info("Initializing Vector Store and RAG Engine for Benchmark...")
    vs = VectorStore()
    rag = RAGEngine(vector_store=vs)

    evaluation_dir = BASE_DIR / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    total_retrieval_time = 0.0
    total_answering_time = 0.0

    # Ensure UTF-8 stdout encoding on Windows
    import sys
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("[ISRO] BHUVANGEO-RAG PERFORMANCE & VERIFICATION BENCHMARK")
    print("=" * 70 + "\n")

    for item in BENCHMARK_TEST_SUITE:
        qid = item["id"]
        q = item["question"]
        cat = item["category"]

        logger.info(f"Evaluating {qid} ({cat}): '{q}'")

        # Measure Retrieval Latency
        t0 = time.time()
        retrieved_chunks = rag.retrieve(query=q, top_k=4)
        t_retrieval = round((time.time() - t0) * 1000, 2)  # ms
        total_retrieval_time += t_retrieval

        # Measure Answer Latency
        t1 = time.time()
        response = rag.answer_question(query=q, top_k=4)
        t_answering = round((time.time() - t1) * 1000, 2)  # ms
        total_answering_time += t_answering

        is_grounded = response.get("is_grounded", False)
        answer = response.get("answer", "")
        citations_count = len(response.get("citations", []))
        top_similarity = response.get("confidence", 0.0)

        # Verification check
        passed_verification = True
        if item["expect_grounded"] and not is_grounded:
            passed_verification = False
        elif not item["expect_grounded"] and is_grounded:
            passed_verification = False

        results.append(
            {
                "id": qid,
                "category": cat,
                "question": q,
                "retrieval_latency_ms": t_retrieval,
                "answering_latency_ms": t_answering,
                "total_latency_ms": round(t_retrieval + t_answering, 2),
                "is_grounded": is_grounded,
                "expected_grounded": item["expect_grounded"],
                "passed_verification": passed_verification,
                "top_similarity": top_similarity,
                "citations_count": citations_count,
                "answer_preview": answer[:180] + "..." if len(answer) > 180 else answer,
            }
        )

        status_icon = "[PASS]" if passed_verification else "[WARN]"
        print(
            f"{status_icon} [{qid}] Latency: {t_retrieval + t_answering:.1f}ms "
            f"(Retr: {t_retrieval:.1f}ms, Ans: {t_answering:.1f}ms) | Grounded: {is_grounded}"
        )

    # Calculate Aggregates
    n_queries = len(BENCHMARK_TEST_SUITE)
    avg_retrieval_ms = round(total_retrieval_time / n_queries, 2)
    avg_answering_ms = round(total_answering_time / n_queries, 2)
    avg_total_ms = round(avg_retrieval_ms + avg_answering_ms, 2)
    accuracy_rate = round(
        sum(1 for r in results if r["passed_verification"]) / n_queries * 100, 2
    )

    report = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries_tested": n_queries,
        "supported_queries_count": 10,
        "anti_hallucination_probes_count": 3,
        "verification_accuracy_percentage": accuracy_rate,
        "latency_metrics": {
            "avg_retrieval_latency_ms": avg_retrieval_ms,
            "avg_answering_latency_ms": avg_answering_ms,
            "avg_end_to_end_latency_ms": avg_total_ms,
        },
        "query_results": results,
    }

    # Save JSON Report
    json_path = evaluation_dir / "performance_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown Summary
    md_path = evaluation_dir / "benchmark_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BhuvanGeo-RAG Performance & Verification Benchmark\n\n")
        f.write(f"- **Total Queries Evaluated:** {n_queries}\n")
        f.write(f"- **Verification / Anti-Hallucination Accuracy:** {accuracy_rate}%\n")
        f.write(f"- **Average Retrieval Latency:** {avg_retrieval_ms} ms\n")
        f.write(f"- **Average Answering Latency:** {avg_answering_ms} ms\n")
        f.write(f"- **Average End-to-End Latency:** {avg_total_ms} ms\n\n")
        f.write("## Test Results Table\n\n")
        f.write(
            "| ID | Category | Retrieval (ms) | Answering (ms) | Total (ms) | Grounded | Verification |\n"
        )
        f.write(
            "|---|---|---|---|---|---|---|\n"
        )
        for r in results:
            icon = "✅ Pass" if r["passed_verification"] else "❌ Fail"
            f.write(
                f"| `{r['id']}` | {r['category']} | {r['retrieval_latency_ms']} | {r['answering_latency_ms']} | {r['total_latency_ms']} | {r['is_grounded']} | {icon} |\n"
            )

    logger.info(f"Performance report saved to {json_path} and {md_path}")
    print("\n" + "=" * 70)
    print(
        f"BENCHMARK COMPLETE: Accuracy: {accuracy_rate}% | Avg Latency: {avg_total_ms}ms"
    )
    print("=" * 70 + "\n")
    return report


if __name__ == "__main__":
    run_benchmark()
