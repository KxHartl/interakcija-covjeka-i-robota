import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glob
import pandas as pd
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

def validate_on_others(classifier, root_dir, exclude_subdirs=None):
    if exclude_subdirs is None:
        exclude_subdirs = []
    all_results = {}
    for dirpath in glob.glob(os.path.join(root_dir, 'norsic-hartl', '*-filtered')):
        if any(esd in dirpath for esd in exclude_subdirs):
            continue
        subject_name = os.path.basename(dirpath)
        result = validate_lda(classifier, dirpath)
        if result is not None:
            metrics = result[0]
            all_results[subject_name] = metrics
    return all_results

if __name__ == "__main__":
    # Priprema lista direktorija s pravim labelama za trening -- koristi pravi / separator!
    prezime = 'norsic-hartl'
    train_brzo = os.path.join('datasets', prezime, prezime+'-brzo-filtered')
    train_sporo = os.path.join('datasets', prezime, prezime+'-sporo-filtered')
    train_dirs = [train_brzo, train_sporo]
    root_dir = 'datasets'
    lda_model, train_data = train_lda(train_dirs)
    if lda_model is None:
        exit(1)
    exclude_dirs = [train_brzo, train_sporo]
    results = validate_on_others(lda_model, root_dir, exclude_subdirs=exclude_dirs)
    print("Rezultati validacije na ostalim skupovima:")
    for subject, metrics in results.items():
        print(f"\n=== {subject} ===")
        for met, val in metrics.items():
            if met != 'report':
                print(f"{met}: {val:.4f}")
        print("Klasifikacijski izvještaj:")
        print(metrics['report'])
