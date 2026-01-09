import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lda_feature_extraction import extract_features_and_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report
)

# -----------------------------
# UTVRĐIVANJE KLASE IZ IMENA MAPE
# -----------------------------
def get_class_from_dirname(dirname):
    """Vraća klasu (0=sporo, 1=brzo) prema nazivu direktorija"""
    d = dirname.lower()
    if "brzo" in d:
        return 1
    if "sporo" in d:
        return 0
    return None


# -----------------------------
# TRENING LDA
# -----------------------------
def train_lda(train_dirs):
    X_train = []
    y_train = []
    train_files = []

    for td in train_dirs:
        Xpart, ypart, fnames = extract_features_and_labels(td)
        # Za svaki direktorij eksplicitno prepiši label prema imenu dir-a
        label = get_class_from_dirname(td)
        if label is None:
            print(f"UPOZORENJE: Ne prepoznajem klasu iz imena direktorija: {td}")
            continue

        ypart = np.full(len(Xpart), label)

        X_train.append(Xpart)
        y_train.append(ypart)
        train_files += fnames

    if not X_train:
        print(f"Nema podataka za treniranje u {train_dirs}")
        return None, ([], [], [])

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    return lda, (X_train, y_train, train_files)


# -----------------------------
# VALIDACIJA NA JEDNOM DIREKTORIJU
# -----------------------------
def validate_lda(classifier, val_dir):
    X_val, y_val, val_files = extract_features_and_labels(val_dir)

    true_label = get_class_from_dirname(val_dir)
    if true_label is None:
        print(f"Ne mogu odrediti klasu iz mape: {val_dir}, preskačem.")
        return None

    # Zamijeni (ignoriraj stare labele iz funkcije) i koristi label iz direktorija
    y_val = np.full(len(X_val), true_label)

    if X_val.shape[0] == 0:
        print(f"Nema validacijskih podataka u {val_dir}")
        return None

    y_pred = classifier.predict(X_val)

    results = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_val, y_pred),
        'report': classification_report(
            y_val, y_pred,
            target_names=['Sporo', 'Brzo'],
            zero_division=0
        )
    }

    return results, y_val, y_pred, val_files


# -----------------------------
# PRONALAZAK SVIH -filtered MAPA OSIM TRENINGA
# -----------------------------
def get_all_filtered_dirs(exclude_dirs):
    all_dirs = []
    for root, dirs, files in os.walk('datasets'):
        for d in dirs:
            if d.endswith("-filtered"):
                full_path = os.path.join(root, d)
                if full_path not in exclude_dirs:
                    all_dirs.append(full_path)
    return all_dirs


# -----------------------------
# VALIDACIJA NA SVIMA OSIM TRENINGA
# -----------------------------
def validate_on_all_filtered_dirs(classifier, exclude_dirs):
    all_results = {}
    all_val_dirs = get_all_filtered_dirs(exclude_dirs)

    for val_dir in all_val_dirs:
        subject_name = os.path.basename(val_dir)
        result = validate_lda(classifier, val_dir)
        if result is not None:
            metrics = result[0]
            all_results[subject_name] = metrics

    return all_results


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    prezime1 = 'norsic'
    prezime2 = 'hartl'
    korijen = f'{prezime1}-{prezime2}'

    root_path = os.path.join('datasets', korijen)

    # Treniramo samo na brzo/sporo mapama za prvo prezime
    train_dirs = [
        os.path.join(root_path, f'{prezime1}-brzo-filtered'),
        os.path.join(root_path, f'{prezime1}-sporo-filtered'),
    ]
    train_dirs = [d for d in train_dirs if os.path.isdir(d)]

    print("Trening set direktoriji:")
    for td in train_dirs:
        print(f"  {td}")

    lda_model, train_data = train_lda(train_dirs)
    if lda_model is None:
        exit(1)

    print("\nValidacija na svim -filtered setovima u cijelom datasets drvetu:")
    results = validate_on_all_filtered_dirs(lda_model, train_dirs)

    for subject, metrics in results.items():
        print(f"\n=== {subject} ===")
        for met, val in metrics.items():
            if met != 'report':
                print(f"{met}: {val:.4f}")
        print("Klasifikacijski izvještaj:")
        print(metrics['report'])
