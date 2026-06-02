import os
import pandas as pd
from sklearn.metrics import classification_report


CLASS_NAMES = ['None', 'Left', 'Right']


def metrics_to_rows(y_true, y_pred, model, train_dataset, test_dataset):
    """
    Convert sklearn classification_report to a list of dicts (one row per class).
    """
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    rows = []
    for cls in CLASS_NAMES:
        rows.append({
            'model':         model,
            'train_dataset': train_dataset,
            'test_dataset':  test_dataset,
            'class':         cls,
            'precision':     round(report[cls]['precision'], 4),
            'recall':        round(report[cls]['recall'], 4),
            'f1':            round(report[cls]['f1-score'], 4),
            'support':       int(report[cls]['support']),
        })

    # Also store macro averages
    rows.append({
        'model':         model,
        'train_dataset': train_dataset,
        'test_dataset':  test_dataset,
        'class':         'macro avg',
        'precision':     round(report['macro avg']['precision'], 4),
        'recall':        round(report['macro avg']['recall'], 4),
        'f1':            round(report['macro avg']['f1-score'], 4),
        'support':       int(report['macro avg']['support']),
    })

    return rows


def save_results(rows, filepath="results/benchmark.csv"):
    """
    Append benchmark rows to CSV. Creates the file if it doesn't exist.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    new_df = pd.DataFrame(rows)

    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        combined = pd.concat([existing, new_df], ignore_index=True)
        # Deduplicate: keep last run for same model/train/test/class combo
        combined = combined.drop_duplicates(
            subset=['model', 'train_dataset', 'test_dataset', 'class'],
            keep='last'
        )
        combined.to_csv(filepath, index=False)
    else:
        new_df.to_csv(filepath, index=False)

    print(f"Benchmark results saved to {filepath}")


def print_benchmark_table(filepath="results/benchmark.csv"):
    """Print a formatted view of the benchmark CSV."""
    if not os.path.exists(filepath):
        print("No benchmark results found.")
        return

    df = pd.read_csv(filepath)
    macro = df[df['class'] == 'macro avg'].copy()
    macro = macro.sort_values(['model', 'train_dataset', 'test_dataset'])

    print("\n=== Benchmark Summary (Macro Avg F1) ===")
    print(macro[['model', 'train_dataset', 'test_dataset', 'precision', 'recall', 'f1']].to_string(index=False))
