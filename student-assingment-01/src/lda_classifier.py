import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glob
import pandas as pd
import numpy as np
from lda_feature_extraction import extract_features_and_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report

def train_lda(train_dir):
    X_train, y_train, train_files = extract_features_and_labels(train_dir)
    if X_train.shape[0] == 0:
        print(f"Nema podatka za treniranje u {train_dir}")
        return None, (X_train, y_train, train_files)
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

def validate_on_others(classifier, root_dir, exclude_subdir=None):
    all_results = {}
    for dirpath in glob.glob(os.path.join(root_dir, '*-filtered')):
        if exclude_subdir and exclude_subdir in dirpath:
            continue
        subject_name = os.path.basename(dirpath)
        result = validate_lda(classifier, dirpath)
        if result is not None:
            metrics = result[0]
            all_results[subject_name] = metrics
    return all_results

if __name__ == "__main__":
    train_dir = 'datasets/norsic_hartl/norsic-brzina-filtered'
    root_dir = 'datasets'
    lda_model, train_data = train_lda(train_dir)
    if lda_model is None:
        exit(1)
    results = validate_on_others(lda_model, root_dir, exclude_subdir='norsic_hartl/norsic-brzina-filtered')
    print("Rezultati validacije na ostalim skupovima:")
    for subject, metrics in results.items():
        print(f"\n=== {subject} ===")
        for met, val in metrics.items():
            if met != 'report':
                print(f"{met}: {val:.4f}")
        print("Klasifikacijski izvještaj:")
        print(metrics['report'])
