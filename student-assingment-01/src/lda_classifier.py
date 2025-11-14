import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report

if __name__ == "__main__":
    # Učitaj spremljeni dataset
    df = pd.read_csv('datasets/normalized/all_measurements.csv')
    print(f"Ukupno mjerenja: {len(df)}")
    print(f"Klase: {df['klasa'].value_counts().sort_index().to_dict()}")

    # Razdvoji X, y (X: brzina, sila, y: klasa)
    X = df[['brzina', 'sila']].values
    y = df['klasa'].values
    prezime = df['prezime'].values

    # Train/test split (može i cross-val, demo ovdje stratificirani split)
    X_train, X_test, y_train, y_test, prezime_train, prezime_test = train_test_split(
        X, y, prezime, test_size=0.33, random_state=42, stratify=y)

    # Treniraj LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    labels = [0,1]
    target_names = ['Sporo','Brzo']
    print(f"\nRezultati testa:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, labels=labels, average=None, zero_division=0)}")
    print(f"Recall: {recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)}")
    print(f"F1: {f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)}")
    print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))
