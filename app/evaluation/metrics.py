from typing import List, Sequence
import numpy as np


def recall_at_k(
    retrieved: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]], k: int
) -> float:
    """Compute Recall@K for a list of queries. retrieved and relevant are lists of lists."""
    if k <= 0:
        return 0.0

    total_recall = 0.0
    n = len(retrieved)

    for retrieved_list, relevant_list in zip(retrieved, relevant):
        topk = set(retrieved_list[:k])
        relevant_set = set(relevant_list)
        if relevant_set:
            total_recall += len(topk & relevant_set) / len(relevant_set)

    return total_recall / max(1, n)


def precision_at_k(
    retrieved: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]], k: int
) -> float:
    """Compute Precision@K for a list of queries."""
    if k <= 0:
        return 0.0

    total_precision = 0.0
    n = len(retrieved)

    for retrieved_list, relevant_list in zip(retrieved, relevant):
        topk = set(retrieved_list[:k])
        relevant_set = set(relevant_list)
        if topk:
            total_precision += len(topk & relevant_set) / min(k, len(retrieved_list))

    return total_precision / max(1, n)


def ndcg_at_k(
    retrieved: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]], k: int
) -> float:
    """Compute NDCG@K for a list of queries."""
    if k <= 0:
        return 0.0

    total_ndcg = 0.0
    n = len(retrieved)

    for retrieved_list, relevant_list in zip(retrieved, relevant):
        if not relevant_list:
            continue

        relevant_set = set(relevant_list)
        dcg = 0.0
        for i, doc in enumerate(retrieved_list[:k], start=1):
            if doc in relevant_set:
                dcg += 1.0 / np.log2(i + 1)

        idcg = sum(
            1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant_set)) + 1)
        )

        if idcg > 0:
            total_ndcg += dcg / idcg

    return total_ndcg / max(1, n)


def mrr(
    retrieved_lists: Sequence[Sequence[str]], relevant_lists: Sequence[Sequence[str]]
) -> float:
    """Mean Reciprocal Rank for a set of queries.

    retrieved_lists and relevant_lists must be same length.
    """
    assert len(retrieved_lists) == len(relevant_lists)
    rr_sum = 0.0
    n = len(retrieved_lists)
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        rank = 0
        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                rank = 1.0 / i
                break
        rr_sum += rank
    return rr_sum / max(1, n)


def grounding_rate(grounding_scores: Sequence[float], threshold: float = 0.5) -> float:
    """Proportion of queries with grounding score >= threshold."""
    if not grounding_scores:
        return 0.0
    ok = sum(1 for s in grounding_scores if s >= threshold)
    return ok / len(grounding_scores)


def hallucination_rate(
    unsupported_counts: Sequence[int], total_claims: Sequence[int]
) -> float:
    """Estimate hallucination rate as ratio of unsupported claims to total claims across queries."""
    total_unsupported = sum(unsupported_counts)
    total = sum(total_claims)
    if total == 0:
        return 0.0
    return total_unsupported / total


def expected_calibration_error(
    confidences: Sequence[float], accuracies: Sequence[float], n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    if not confidences or not accuracies:
        return 0.0

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (np.sum(in_bin) / len(confidences)) * abs(bin_acc - bin_conf)

    return ece


def pearson_corr(confidences: Sequence[float], accuracies: Sequence[float]) -> float:
    """Compute Pearson correlation between confidence and accuracy."""
    if not confidences or not accuracies or len(confidences) != len(accuracies):
        return 0.0

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    mean_conf = np.mean(confidences)
    mean_acc = np.mean(accuracies)

    cov = np.sum((confidences - mean_conf) * (accuracies - mean_acc))
    std_conf = np.sqrt(np.sum((confidences - mean_conf) ** 2))
    std_acc = np.sqrt(np.sum((accuracies - mean_acc) ** 2))

    if std_conf == 0 or std_acc == 0:
        return 0.0

    return cov / (std_conf * std_acc)
