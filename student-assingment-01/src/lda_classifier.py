import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glob
import numpy as np
from lda_feature_extraction import extract_features_and_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
import matplotlib.pyplot as plt

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
    # U train_lda funkciji, nakon fit-a:
    print(f"\nTrening set značajke:")
    print(f"  X shape: {X_train.shape}")
    print(f"  y values: {np.unique(y_train, return_counts=True)}")
    print(f"  Prosječna brzina za sporo (0): {X_train[y_train==0, 0].mean():.6f}")
    print(f"  Prosječna brzina za brzo (1): {X_train[y_train==1, 0].mean():.6f}")
    print(f"  Prosječna sila za sporo (0): {X_train[y_train==0, 1].mean():.6f}")
    print(f"  Prosječna sila za brzo (1): {X_train[y_train==1, 1].mean():.6f}")
    return lda, (X_train, y_train, train_files)

def validate_lda(classifier, val_dir):
    X_val, y_val, val_files = extract_features_and_labels(val_dir)
    if X_val.shape[0] == 0:
        print(f"Nema validacijskih podataka u {val_dir}")
        return None
    y_pred = classifier.predict(X_val)
    # Dinamički utvrdi koji labeli su zaista prisutni
    unique_labels = np.unique(np.concatenate([y_val, y_pred]))
    # Osiguraj uvijek poznate klase (0 i 1) ako ih ima
    all_labels = [0, 1]
    target_names = ["Sporo", "Brzo"]
    results = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, labels=all_labels, zero_division=0, average=None),
        'recall': recall_score(y_val, y_pred, labels=all_labels, zero_division=0, average=None),
        'f1': f1_score(y_val, y_pred, labels=all_labels, zero_division=0, average=None),
        'mcc': matthews_corrcoef(y_val, y_pred),
        'report': classification_report(y_val, y_pred, labels=all_labels, target_names=target_names, zero_division=0)
    }
    return results, y_val, y_pred, val_files

def get_all_filtered_dirs(exclude_dirs):
    all_dirs = []
    for root, dirs, files in os.walk('datasets'):
        for d in dirs:
            if d.endswith('-filtered'):
                full_path = os.path.join(root, d)
                if full_path not in exclude_dirs:
                    all_dirs.append(full_path)
    return all_dirs

def validate_on_all_filtered_dirs(classifier, exclude_dirs):
    all_results = {}
    all_val_dirs = get_all_filtered_dirs(exclude_dirs)
    for val_dir in all_val_dirs:
        label = None
        if 'brzo-filtered' in os.path.basename(val_dir):
            label = 'brzo'
        elif 'sporo-filtered' in os.path.basename(val_dir):
            label = 'sporo'
        subject_name = os.path.basename(val_dir)
        result = validate_lda(classifier, val_dir)
        if result is not None:
            metrics = result[0]
            all_results[subject_name] = metrics
    return all_results

if __name__ == "__main__":
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
    
    X_train, y_train, _ = train_data

    import matplotlib.pyplot as plt

    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', label='Sporo')
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', label='Brzo')
    plt.xlabel('Prosječna brzina')
    plt.ylabel('Prosječna sila')
    plt.legend()
    plt.title("Scatter plot za trening podatke: Prosječna brzina vs sila")
    plt.show()

    print("\nValidacija na svim -filtered setovima u cijelom datasets drvetu:")
    results = validate_on_all_filtered_dirs(lda_model, train_dirs)
    for subject, metrics in results.items():
        print(f"\n=== {subject} ===")
        for met, val in metrics.items():
            if met != 'report':
                print(f"{met}: {val}")
        print("Klasifikacijski izvještaj:")
        print(metrics['report'])
