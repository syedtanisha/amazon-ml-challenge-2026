import pandas as pd


def evaluate_macro_f05(
    ground_truth_map: dict[str, set[str]],
    predictions_map: dict[str, set[str]],
    s1_ids: list[str],
) -> tuple[float, float, float]:
    """
    Compute macro-averaged Precision, Recall, and F0.5 across Source 1 entities.

    F0.5 weighting:
        F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R)
             = 1.25 * P * R / (0.25 * P + R)

    Special cases per S1 entity:
        - True singleton (T is empty) and predicted singleton (P is empty):
          Precision = 1.0, Recall = 1.0, F0.5 = 1.0
        - True singleton (T is empty) and predicted non-singleton (P non-empty):
          Precision = 0.0, Recall = 0.0, F0.5 = 0.0
        - True non-singleton (T non-empty) and predicted singleton (P empty):
          Precision = 0.0, Recall = 0.0, F0.5 = 0.0
        - True non-singleton (T non-empty) and predicted non-singleton (P non-empty):
          TP = |P & T|, FP = |P - T|, FN = |T - P|
          Precision = TP / (TP + FP)
          Recall = TP / (TP + FN)
          F0.5 = 1.25 * P * R / (0.25 * P + R) if (P + R > 0) else 0.0
    """

    sum_p = 0.0
    sum_r = 0.0
    sum_f05 = 0.0
    count = len(s1_ids)

    if count == 0:
        return 0.0, 0.0, 0.0

    for s1_id in s1_ids:
        t = ground_truth_map.get(s1_id, set())
        p = predictions_map.get(s1_id, set())

        if not t and not p:
            prec, rec, f05 = 1.0, 1.0, 1.0
        elif not t and p:
            prec, rec, f05 = 0.0, 0.0, 0.0
        elif t and not p:
            prec, rec, f05 = 0.0, 0.0, 0.0
        else:
            tp = len(p & t)
            prec = tp / len(p) if len(p) > 0 else 0.0
            rec = tp / len(t) if len(t) > 0 else 0.0

            if prec + rec > 0:
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            else:
                f05 = 0.0

        sum_p += prec
        sum_r += rec
        sum_f05 += f05

    return sum_p / count, sum_r / count, sum_f05 / count
