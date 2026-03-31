"""
Evaluation Pipeline for Patent Search Engine
=============================================
Generates training/evaluation data (positive & negative pairs) from the patent
corpus and measures retrieval quality with standard IR metrics.

Positive pair strategies:
  1. Claim-to-patent: a claim should retrieve its own patent highly.
  2. Same-classification: patents sharing a classification prefix are related.
  3. Title-to-abstract: a patent's title should retrieve its own abstract.

Negative pair strategies (hard negatives):
  1. Cross-classification: pair claims from different classification families.
  2. Random: randomly sample unrelated claim pairs.

Metrics:
  - Mean Reciprocal Rank (MRR)
  - Recall@K (R@1, R@5, R@10)
  - Mean Average Precision (MAP)
  - Precision@K
"""

import json, time, random
from typing import List, Dict, Tuple
from patent_engine import PatentSearchEngine, Patent, SearchResult


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_positive_pairs(engine: PatentSearchEngine,
                            strategy: str = "claim_to_patent",
                            max_pairs: int = 200) -> List[Dict]:
    """
    Generate (query, relevant_doc_number) positive pairs.

    Strategies:
      claim_to_patent  – query=claim text, relevant=parent patent
      title_to_patent  – query=title, relevant=same patent
      same_class       – query=patent_A abstract, relevant=patent_B (same class)
    """
    pairs = []

    if strategy == "claim_to_patent":
        for p in engine.patents:
            for ci, claim in enumerate(p.claims):
                if len(claim.strip()) < 30:
                    continue
                pairs.append({
                    "query": claim,
                    "relevant_doc": p.doc_number,
                    "type": "positive",
                    "strategy": "claim_to_patent",
                    "meta": {"claim_idx": ci},
                })
                if len(pairs) >= max_pairs:
                    return pairs

    elif strategy == "title_to_patent":
        for p in engine.patents:
            if p.title and p.abstract:
                pairs.append({
                    "query": p.title,
                    "relevant_doc": p.doc_number,
                    "type": "positive",
                    "strategy": "title_to_patent",
                })

    elif strategy == "same_class":
        by_class: Dict[str, List[Patent]] = {}
        for p in engine.patents:
            prefix = p.classification[:4] if len(p.classification) >= 4 else "OTHER"
            by_class.setdefault(prefix, []).append(p)
        for prefix, patents in by_class.items():
            if len(patents) < 2:
                continue
            for i in range(len(patents)):
                for j in range(i + 1, len(patents)):
                    pairs.append({
                        "query": patents[i].abstract or patents[i].title,
                        "relevant_doc": patents[j].doc_number,
                        "type": "positive",
                        "strategy": "same_class",
                        "meta": {"class": prefix},
                    })
                    if len(pairs) >= max_pairs:
                        return pairs

    return pairs[:max_pairs]


def generate_negative_pairs(engine: PatentSearchEngine,
                            strategy: str = "cross_class",
                            max_pairs: int = 200) -> List[Dict]:
    """
    Generate (query, irrelevant_doc_number) negative pairs (hard negatives).
    """
    pairs = []

    if strategy == "cross_class":
        by_class: Dict[str, List[Patent]] = {}
        for p in engine.patents:
            prefix = p.classification[:4] if len(p.classification) >= 4 else "OTHER"
            by_class.setdefault(prefix, []).append(p)
        classes = list(by_class.keys())
        if len(classes) < 2:
            return []
        for c1 in classes:
            for c2 in classes:
                if c1 == c2:
                    continue
                for p1 in by_class[c1][:5]:
                    for p2 in by_class[c2][:3]:
                        # Use a claim from p1 as query, p2 as negative
                        if p1.claims:
                            pairs.append({
                                "query": p1.claims[0],
                                "relevant_doc": p2.doc_number,
                                "type": "negative",
                                "strategy": "cross_class",
                                "meta": {"query_class": c1, "doc_class": c2},
                            })
                            if len(pairs) >= max_pairs:
                                return pairs

    elif strategy == "random":
        all_docs = list(engine.patents)
        for _ in range(max_pairs):
            p1, p2 = random.sample(all_docs, 2)
            if p1.classification[:4] == p2.classification[:4]:
                continue  # skip same-class (not a true negative)
            if p1.claims:
                pairs.append({
                    "query": random.choice(p1.claims),
                    "relevant_doc": p2.doc_number,
                    "type": "negative",
                    "strategy": "random",
                })

    return pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    engine: PatentSearchEngine,
    pairs: List[Dict],
    level: str = "patent",
    top_k: int = 10,
    method: str = "combined",
) -> Dict:
    """
    Evaluate retrieval quality on a set of (query, relevant_doc) pairs.

    Returns metrics: MRR, R@1, R@5, R@10, MAP, avg_score_positive, avg_score_negative.
    """
    mrr_sum = 0.0
    recall_at = {1: 0, 5: 0, 10: 0}
    ap_sum = 0.0
    pos_scores = []
    neg_scores = []
    total = len(pairs)
    t0 = time.perf_counter()

    for pair in pairs:
        query = pair["query"]
        target_doc = pair["relevant_doc"]
        is_positive = pair["type"] == "positive"

        results, _ = engine.search(query, level=level, top_k=top_k, method=method)
        retrieved_docs = [r.patent.doc_number for r in results]
        retrieved_scores = [r.score for r in results]

        if is_positive:
            # Find rank of relevant doc
            rank = None
            for i, doc in enumerate(retrieved_docs):
                if doc == target_doc:
                    rank = i + 1
                    break

            if rank is not None:
                mrr_sum += 1.0 / rank
                for k in recall_at:
                    if rank <= k:
                        recall_at[k] += 1
                # AP: precision at the rank where the relevant doc appears
                ap_sum += 1.0 / rank
                pos_scores.append(retrieved_scores[rank - 1] if rank <= len(retrieved_scores) else 0.0)
            else:
                pos_scores.append(0.0)
        else:
            # For negatives: we want the irrelevant doc to NOT be retrieved highly
            if target_doc in retrieved_docs:
                idx = retrieved_docs.index(target_doc)
                neg_scores.append(retrieved_scores[idx])
            else:
                neg_scores.append(0.0)

    elapsed = time.perf_counter() - t0
    n_pos = sum(1 for p in pairs if p["type"] == "positive")
    n_neg = total - n_pos

    metrics = {
        "total_pairs": total,
        "positive_pairs": n_pos,
        "negative_pairs": n_neg,
        "method": method,
        "level": level,
        "eval_time_sec": round(elapsed, 3),
    }

    if n_pos > 0:
        metrics["MRR"] = round(mrr_sum / n_pos, 4)
        metrics["Recall@1"] = round(recall_at[1] / n_pos, 4)
        metrics["Recall@5"] = round(recall_at[5] / n_pos, 4)
        metrics["Recall@10"] = round(recall_at[10] / n_pos, 4)
        metrics["MAP"] = round(ap_sum / n_pos, 4)
        metrics["avg_positive_score"] = round(sum(pos_scores) / len(pos_scores), 4) if pos_scores else 0.0

    if n_neg > 0:
        metrics["avg_negative_score"] = round(sum(neg_scores) / len(neg_scores), 4) if neg_scores else 0.0
        # Separation: how well positives and negatives are separated
        if pos_scores and neg_scores:
            metrics["score_separation"] = round(
                (sum(pos_scores)/len(pos_scores)) - (sum(neg_scores)/len(neg_scores)), 4
            )

    return metrics


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def run_full_evaluation(engine: PatentSearchEngine) -> Dict:
    """Run a comprehensive evaluation across multiple strategies and methods."""
    print("\n" + "=" * 60)
    print("PATENT SEARCH ENGINE — EVALUATION REPORT")
    print("=" * 60)

    all_results = {}

    # Generate pairs
    print("\n[1] Generating evaluation pairs...")
    pos_claim = generate_positive_pairs(engine, "claim_to_patent", max_pairs=100)
    pos_title = generate_positive_pairs(engine, "title_to_patent", max_pairs=40)
    pos_class = generate_positive_pairs(engine, "same_class", max_pairs=60)
    neg_cross = generate_negative_pairs(engine, "cross_class", max_pairs=80)
    neg_rand = generate_negative_pairs(engine, "random", max_pairs=40)

    print(f"  Positive pairs: {len(pos_claim)} claim→patent, "
          f"{len(pos_title)} title→patent, {len(pos_class)} same-class")
    print(f"  Negative pairs: {len(neg_cross)} cross-class, {len(neg_rand)} random")

    # Evaluate claim-to-patent retrieval
    print("\n[2] Evaluating claim-to-patent retrieval...")
    for method in ["tfidf", "bm25", "combined"]:
        combined = pos_claim + neg_cross[:len(pos_claim)]
        metrics = evaluate_retrieval(engine, combined, level="patent", method=method)
        key = f"claim_to_patent_{method}"
        all_results[key] = metrics
        print(f"  {method:>8}: MRR={metrics.get('MRR','N/A'):.4f}  "
              f"R@1={metrics.get('Recall@1','N/A'):.4f}  "
              f"R@5={metrics.get('Recall@5','N/A'):.4f}  "
              f"R@10={metrics.get('Recall@10','N/A'):.4f}  "
              f"sep={metrics.get('score_separation','N/A')}")

    # Evaluate title-to-patent retrieval
    print("\n[3] Evaluating title-to-patent retrieval...")
    for method in ["tfidf", "bm25", "combined"]:
        metrics = evaluate_retrieval(engine, pos_title, level="patent", method=method)
        key = f"title_to_patent_{method}"
        all_results[key] = metrics
        print(f"  {method:>8}: MRR={metrics.get('MRR','N/A'):.4f}  "
              f"R@1={metrics.get('Recall@1','N/A'):.4f}  "
              f"R@5={metrics.get('Recall@5','N/A'):.4f}")

    # Compare with/without hybrid filtering
    print("\n[4] Hybrid search timing comparison...")
    query = "non-pneumatic tire with deformable structure"
    for label, kwargs in [
        ("No filter", {}),
        ("Class=B60C", {"classification_prefix": "B60C"}),
        ("Keyword='tire'", {"keywords": ["tire"]}),
        ("Class+Keyword", {"classification_prefix": "B60C", "keywords": ["tire"]}),
    ]:
        _, timing = engine.search(query, level="patent", top_k=10, **kwargs)
        print(f"  {label:<20}: {timing['total_ms']:.2f}ms  "
              f"({timing['results_returned']} results)")
        all_results[f"timing_{label}"] = timing

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    return all_results


# ---------------------------------------------------------------------------
# Export pairs for potential fine-tuning
# ---------------------------------------------------------------------------

def export_training_data(engine: PatentSearchEngine, output_path: str = "training_pairs.json"):
    """Export all generated pairs as a JSON file for fine-tuning."""
    pos = (generate_positive_pairs(engine, "claim_to_patent", 200) +
           generate_positive_pairs(engine, "title_to_patent", 50) +
           generate_positive_pairs(engine, "same_class", 100))
    neg = (generate_negative_pairs(engine, "cross_class", 150) +
           generate_negative_pairs(engine, "random", 50))

    data = {"positive_pairs": pos, "negative_pairs": neg,
            "total": len(pos) + len(neg)}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported {len(pos)} positive + {len(neg)} negative pairs → {output_path}")
    return data


if __name__ == "__main__":
    from patent_engine import create_engine
    engine = create_engine("data/patent_data_small")
    results = run_full_evaluation(engine)
    export_training_data(engine)
    print("\nAll results:")
    print(json.dumps({k: v for k, v in results.items() if isinstance(v, dict)}, indent=2))
