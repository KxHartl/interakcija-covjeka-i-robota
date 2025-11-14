import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report

TRAIN_PREZIMENA = ['norsic', 'hartl']  # OVDJE NAVEDI PREZIMENA ZA TRENING (npr. ['norsic'] ili ['cuk','gugo'])

if __name__ == "__main__":
    df = pd.read_csv('datasets/normalized/all_measurements.csv')
    print(f"Ukupno mjerenja: {len(df)}")
    print(f"Kandidati: {sorted(df['prezime'].unique())}")
    print(f"Trening prezimena: {TRAIN_PREZIMENA}")
    
    # Trening podaci
    df_train = df[df['prezime'].isin(TRAIN_PREZIMENA)]
    df_val = df[~df['prezime'].isin(TRAIN_PREZIMENA)]

    print(f"Trening set veličina: {len(df_train)}, Validacija: {len(df_val)}")
    print(f"Klase u treningu: {df_train['klasa'].value_counts().to_dict()}")
    print(f"Klase u validaciji: {df_val['klasa'].value_counts().to_dict()}")

    X_train, y_train = df_train[['brzina', 'sila']].values, df_train['klasa'].values
    X_val, y_val, prezime_val = df_val[['brzina', 'sila']].values, df_val['klasa'].values, df_val['prezime'].values

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_val)

    labels = [0,1]
    target_names = ['Sporo','Brzo']

    # Agregatno po cijeloj validaciji
    print(f"\nVALIDACIJA na svima koji nisu {TRAIN_PREZIMENA}:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, labels=labels, average=None, zero_division=0)}")
    print(f"Recall: {recall_score(y_val, y_pred, labels=labels, average=None, zero_division=0)}")
    print(f"F1: {f1_score(y_val, y_pred, labels=labels, average=None, zero_division=0)}")
    print(f"MCC: {matthews_corrcoef(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred, labels=labels, target_names=target_names, zero_division=0))

    # Po svakom prezimenu posebno
    for prezime in sorted(set(prezime_val)):
        idx = (prezime_val == prezime)
        print(f"\n=== {prezime} ===")
        if idx.sum()==0:
            print("(nema podataka)")
            continue
        y_val_p = y_val[idx]
        y_pred_p = y_pred[idx]
        print(f"Accuracy: {accuracy_score(y_val_p, y_pred_p):.4f}")
        print(f"Precision: {precision_score(y_val_p, y_pred_p, labels=labels, average=None, zero_division=0)}")
        print(f"Recall: {recall_score(y_val_p, y_pred_p, labels=labels, average=None, zero_division=0)}")
        print(f"F1: {f1_score(y_val_p, y_pred_p, labels=labels, average=None, zero_division=0)}")
        print(f"MCC: {matthews_corrcoef(y_val_p, y_pred_p):.4f}")
        print(classification_report(y_val_p, y_pred_p, labels=labels, target_names=target_names, zero_division=0))
