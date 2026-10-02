from pathlib import Path
from lab2_starter import load_predictions
import numpy as np
import matplotlib.pyplot as plt
data = load_predictions(Path("results"))
y_validation = data["validation_y"]
s_validation = data["validation_scores"]
y_test = data["test_y"]
s_test = data["test_scores"]
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
import pandas as pd
def plot_validation_curves(y_validation, s_validation):
    precision, recall, _ = precision_recall_curve(
        y_validation, s_validation
    )
    fpr, tpr, _ = roc_curve(y_validation, s_validation)
    roc_auc = roc_auc_score(y_validation, s_validation)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve")
    plt.savefig("results/pr_curve.png")
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.savefig("results/roc_curve.png")
    plt.close()
def cumulative_threshold_search(d_true, d_scores):
    """
    Compute the confusion matrix for a binary classification task.

    
    Parameters:
    d_true (np.ndarray): True binary labels (0 or 1).
    d_scores (np.ndarray): Predicted scores (continuous values).
   
    
    Returns:
    candidates (pd.DataFrame): DataFrame containing thresholds and corresponding confusion matrix values.
    t_star (float): Threshold that minimizes the cost function C = 10*FN + FP.
    """


    order = np.argsort(-d_scores)  
    d_scores_sorted = d_scores[order]
    d_true_sorted = d_true[order]

    total_positives = np.sum(d_true_sorted == 1)
    total_negatives = np.sum(d_true_sorted == 0)

    TP = 0
    FP = 0
    FN = total_positives
    TN = total_negatives
    C = 10 * FN + FP

    rows = [{
    "threshold": np.inf,
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "TN": TN,
    "C": C,
    }]

    i = 0
    n = len(d_scores_sorted)

    while i < n:
        t = d_scores_sorted[i]
        j = i + 1

        while j < n and d_scores_sorted[j] == t:
            j += 1

        group_y = d_true_sorted[i:j]
        TP += np.sum(group_y == 1)
        FP += np.sum(group_y == 0)

        FN = total_positives - TP
        TN = total_negatives - FP
        C = 10 * FN + FP
        rows.append({
        "threshold": t,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "C": C,
        })

        i = j
    candidates = pd.DataFrame(rows)

    min_cost = candidates["C"].min()
    best = (
        candidates[candidates["C"] == min_cost]
        .sort_values("threshold", ascending=False)
        .iloc[0]
    )

    t_star = best["threshold"]
    
    return  candidates, t_star    

def metrics_report(d_true, d_scores, t):
    """
    Compute the confusion matrix for a binary classification task.

    Parameters:
    d_true (np.ndarray): True binary labels (0 or 1).
    d_scores (np.ndarray): Predicted scores (continuous values).
    t (float): Threshold for classification.

    Returns:
    d_marked (np.ndarray): Array of strings indicating TP, FP, FN, TN for each instance.
    cm (np.ndarray): Confusion matrix as a 2x2 array.
    precision (float): Precision of the predictions.
    recall (float): Recall of the predictions.
    C (float): Cost calculated as 10*FN + FP.
    """
    predictions = d_scores >= t
    TP = np.sum((predictions == 1) & (d_true == 1))
    FP = np.sum((predictions == 1) & (d_true == 0))
    FN = np.sum((predictions == 0) & (d_true == 1))
    TN = np.sum((predictions == 0) & (d_true == 0))
    C = 10 * FN + FP

    d_marked = np.where((predictions == 1) & (d_true == 1), "TP",
                np.where((predictions == 1) & (d_true == 0), "FP",
                np.where((predictions == 0) & (d_true == 1), "FN", "TN")))
    
     
    cm = np.array([
    [TN, FP],
    [FN, TP],
    ])
    precision = TP / (TP + FP) if (TP + FP) > 0 else None
    recall = TP / (TP + FN) if (TP + FN) > 0 else None
    
    return d_marked, cm, precision, recall, C

def plot_confusion_heatmap(cm, path, title):
    

    fig, ax = plt.subplots()
    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1], labels=["0: легітимна", "1: шахрайство"])
    ax.set_yticks([0, 1], labels=["0: легітимна", "1: шахрайство"])
    ax.set_xlabel("Прогноз")
    ax.set_ylabel("Справжня мітка")
    ax.set_title(title)

    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center")

    plt.colorbar(image, ax=ax)
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def brute_force_search(scores, y):
    scores = np.asarray(scores)
    y = np.asarray(y)

    # Усі унікальні оцінки та +inf, від більшого порога до меншого
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    rows = []

    for t in thresholds:
        predicted = scores >= t

        TP = np.count_nonzero(predicted & (y == 1))
        FP = np.count_nonzero(predicted & (y == 0))
        FN = np.count_nonzero(~predicted & (y == 1))
        TN = np.count_nonzero(~predicted & (y == 0))
        C = 10 * FN + FP

        rows.append({
            "threshold": t,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "C": C,
        })

    candidates = pd.DataFrame(rows)

    # За однакової мінімальної вартості обираємо найбільший поріг
    min_cost = candidates["C"].min()
    best = (
        candidates[candidates["C"] == min_cost]
        .sort_values("threshold", ascending=False)
        .iloc[0]
    )

    return candidates, best
            
A_scores = np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1])
A_y      = np.array([0,   1,   0,   1,   0,   0])

B_scores = np.array([0.9, 0.6, 0.3])
B_y      = np.array([0,   0,   0])

C_scores = np.array([0.8] + [0.5] * 11 + [0.2])
C_y      = np.array([1, 1] + [0] * 11)


plot_validation_curves(y_validation, s_validation)

fast_candidates_validation, t_star_validation = (
    cumulative_threshold_search(y_validation, s_validation)
)
fast_candidates_validation.to_csv("fast_candidates_validation.csv", index=False)
fast_candidates_A, fast_t_star_A = cumulative_threshold_search(A_y, A_scores)
fast_candidates_B, fast_t_star_B = cumulative_threshold_search(B_y, B_scores)
fast_candidates_C, fast_t_star_C = cumulative_threshold_search(C_y, C_scores)

print("Candidates validation:\n", fast_candidates_validation)
print("Best threshold validation:", t_star_validation)
print("Candidates A:\n", fast_candidates_A)
print("Best threshold A:", fast_t_star_A)
print("Candidates B:\n", fast_candidates_B)
print("Best threshold B:", fast_t_star_B)
print("Candidates C:\n", fast_candidates_C)
print("Best threshold C:", fast_t_star_C)

brute_candidates_A, brute_t_star_A = brute_force_search(A_scores, A_y)
brute_candidates_B, brute_t_star_B = brute_force_search(B_scores, B_y)
brute_candidates_C, brute_t_star_C = brute_force_search(C_scores, C_y)


print("Brute force candidates A:\n", brute_candidates_A)
print("Brute force best threshold A:", brute_t_star_A)
print("Brute force candidates B:\n", brute_candidates_B)
print("Brute force best threshold B:", brute_t_star_B)
print("Brute force candidates C:\n", brute_candidates_C)
print("Brute force best threshold C:", brute_t_star_C)

pd.testing.assert_frame_equal(
    fast_candidates_A.reset_index(drop=True),
    brute_candidates_A.reset_index(drop=True),
    check_dtype=False,
)
assert fast_t_star_A == brute_t_star_A["threshold"]

pd.testing.assert_frame_equal(
    fast_candidates_B.reset_index(drop=True),
    brute_candidates_B.reset_index(drop=True),
    check_dtype=False,
)
assert fast_t_star_B == brute_t_star_B["threshold"]

pd.testing.assert_frame_equal(
    fast_candidates_C.reset_index(drop=True),
    brute_candidates_C.reset_index(drop=True),
    check_dtype=False,
)
assert fast_t_star_C == brute_t_star_C["threshold"]


test_marked, test_cm, test_precision, test_recall, test_C = metrics_report(y_test, s_test, t_star_validation)  # поріг, знайдений на validation
print("Test precision:", test_precision)
print("Test recall:", test_recall)
print("Test cost:", test_C)
plot_confusion_heatmap(test_cm, "confusion_matrix_test.png", "Confusion Matrix - Test Set")
test_errors = pd.DataFrame({
    "CSV row": data["test_row_ids"].astype(int),
    "Type": test_marked,
    "Label": y_test,
    "Score": s_test,
    "Amount": data["test_amount"],
    "Distance from threshold": np.abs(
        data["test_scores"] - t_star_validation
    ),
})

test_errors.to_csv("test_errors.csv", index=False)
fp = (
    test_errors[test_errors["Type"] == "FP"]
    .sort_values("CSV row")
    .head(5)
)
fn = (
    test_errors[test_errors["Type"] == "FN"]
    .sort_values("CSV row")
    .head(5)
)

print("First FP:")
print(fp.to_string(index=False))

print("\nFirst FN:")
print(fn.to_string(index=False))