import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glob
import numpy as np
from lda_feature_extraction import extract_features_and_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report

def train_lda(train_dirs):
    X_train = []
    y_train = []
    train_files = []
    for td in train_dirs:
        Xpart, ypart, fnames = extract_features_and_labels(td)
        X_train.append(Xpart)
        y_train.append(ypart)
        train_files += fnames
    if len(X_train) == 0 or np.concatenate(X_train).shape[0] == 0:
        print(f"Nema podatka za treniranje u {train_dirs}")
        return None, ([], [], [])
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return lda, (X_train, y_train, train_files)

def validate_lda(classifier, val_dir):
    X_val, y_val, val_files = extract_features_and_labels(val_dir)
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
        'report': classification_report(y_val, y_pred, target_names=['Sporo', 'Brzo'])
    }
    return results, y_val, y_pred, val_files

def validate_on_others(classifier, prezime1, prezime2):
    all_results = {}
    # Traži sve kombinacije prezime1-prezime2 kao korijen
    roots = os.path.join('datasets')
    for root in roots:
        for label in ['brzo', 'sporo']:
            filtered_dirs = glob.glob(os.path.join(root, f'*-{label}-filtered'))
            for val_dir in filtered_dirs:
                subject_name = os.path.basename(val_dir)
                # Nemoj validirati na tren setovima
                if val_dir in train_dirs: continue
                result = validate_lda(classifier, val_dir)
                if result is not None:
                    metrics = result[0]
                    all_results[subject_name] = metrics
    return all_results

if __name__ == "__main__":
    # Parametri
    prezime1 = 'norsic'
    prezime2 = 'hartl'
    korijen = f'{prezime1}-{prezime2}'
    root_path = os.path.join('datasets', korijen)
    train_dirs = [os.path.join(root_path, f'{prezime1}-brzo-filtered'), os.path.join(root_path, f'{prezime1}-sporo-filtered')]
    train_dirs = [d for d in train_dirs if os.path.isdir(d)]
    print("Trening set direktoriji:")
    for td in train_dirs:
        print(f"  {td}")
    lda_model, train_data = train_lda(train_dirs)
    if lda_model is None:
        exit(1)
    print("\nValidacija na svim prezime-label-filtered setovima (svaka podmreža iz rootova):")
    results = validate_on_others(lda_model, prezime1, prezime2)
    for subject, metrics in results.items():
        print(f"\n=== {subject} ===")
        for met, val in metrics.items():
            if met != 'report':
                print(f"{met}: {val:.4f}")
        print("Klasifikacijski izvještaj:")
        print(metrics['report'])
