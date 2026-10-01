"""Exact binary ranking metrics with bounded temporary arrays.

One global score sort preserves pooled point metrics. Only the sorted index is
full-sized; threshold counts are streamed, including ties across chunk edges.
"""
import numpy as np


def point_metrics(labels, scores, chunk_size=262144):
    if len(labels) != len(scores) or not len(labels) or chunk_size < 1:
        raise ValueError('Nonempty equal-length arrays and positive chunk size required')
    positives = 0
    for start in range(0, len(labels), chunk_size):
        y, s = labels[start:start+chunk_size], scores[start:start+chunk_size]
        if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
            raise ValueError('Finite scores and binary labels required')
        positives += int(np.count_nonzero(y))
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError('Both point labels required')
    order = np.argsort(scores, kind='quicksort')[::-1]
    seen_positive = 0
    previous_tp = previous_fp = auc_area = ap_area = 0.0
    for start in range(0, len(order), chunk_size):
        ids = order[start:start+chunk_size]
        s, y = scores[ids], labels[ids]
        cumulative = np.cumsum(y, dtype=np.int64) + seen_positive
        ends = np.flatnonzero(s[:-1] != s[1:])
        stop = start + len(ids)
        # Complete the last tie only if it does not continue into the next block.
        if stop == len(order) or s[-1] != scores[order[stop]]:
            ends = np.r_[ends, len(ids)-1]
        if len(ends):
            tp = cumulative[ends].astype(np.float64)
            fp = (start + ends + 1).astype(np.float64) - tp
            old_tp = np.r_[previous_tp, tp[:-1]]
            old_fp = np.r_[previous_fp, fp[:-1]]
            auc_area += float(np.sum((fp-old_fp) * (tp+old_tp) * .5))
            ap_area += float(np.sum((tp-old_tp) * tp / (tp+fp)))
            previous_tp, previous_fp = tp[-1], fp[-1]
        seen_positive = int(cumulative[-1])
    return {'p_auroc': auc_area/(positives*negatives), 'p_ap': ap_area/positives}
