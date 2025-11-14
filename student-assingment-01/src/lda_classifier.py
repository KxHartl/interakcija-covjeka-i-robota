import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report

TRAIN_PREZIMENA = ['norsic', 'hartl']  # OVDJE NAVEDI PREZIMENA ZA TRENING

if __name__ == "__main__":
    df = pd.read_csv('datasets/normalized/all_measurements.csv')
    print(f"Ukupno mjerenja: {len(df)}")
    print(f"Kandidati: {sorted(df['prezime'].unique())}")
    print(f"Trening prezimena: {TRAIN_PREZIMENA}")

    df_train = df[df['prezime'].isin(TRAIN_PREZIMENA)]
    df_val = df[~df['prezime'].isin(TRAIN_PREZIMENA)]
    X_train, y_train = df_train[['brzina', 'sila']].values, df_train['klasa'].values
    X_val, y_val, prezime_val = df_val[['brzina', 'sila']].values, df_val['klasa'].values, df_val['prezime'].values

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_val)

    labels = [0,1]
    target_names = ['Sporo','Brzo']
    # --- Grafički prikaz: prikaz uspjeha po prezimenu ---
    results_summary = []
    prezimena = sorted(set(prezime_val))
    for prezime in prezimena:
        idx = (prezime_val == prezime)
        if idx.sum()==0:
            continue
        y_val_p = y_val[idx]
        y_pred_p = y_pred[idx]
        acc = accuracy_score(y_val_p, y_pred_p)
        prec = precision_score(y_val_p, y_pred_p, labels=labels, average=None, zero_division=0)
        rec = recall_score(y_val_p, y_pred_p, labels=labels, average=None, zero_division=0)
        f1s = f1_score(y_val_p, y_pred_p, labels=labels, average=None, zero_division=0)
        mcc = matthews_corrcoef(y_val_p, y_pred_p)
        results_summary.append({
            'prezime': prezime,
            'accuracy': acc,
            'precision_sporo': prec[0],
            'precision_brzo': prec[1],
            'recall_sporo': rec[0],
            'recall_brzo': rec[1],
            'f1_sporo': f1s[0],
            'f1_brzo': f1s[1],
            'mcc': mcc
        })
    res_df = pd.DataFrame(results_summary)
    # Barplot za accuracy po prezimenu
    if not res_df.empty:
        plt.figure(figsize=(9,6))
        plt.bar(res_df['prezime'], res_df['accuracy'], color='royalblue')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.xlabel('Prezime (validacija)')
        plt.title(f'Accuracy na svako prezime (Test na {TRAIN_PREZIMENA})')
        plt.tight_layout()
        plt.show()
        # Heatmap za ostale metrike
        res_hm = res_df.set_index('prezime')[['precision_sporo','precision_brzo','recall_sporo','recall_brzo','f1_sporo','f1_brzo','mcc']]
        plt.figure(figsize=(12,6))
        import seaborn as sns
        sns.heatmap(res_hm, annot=True, fmt='.2f', cmap='YlGnBu', center=0.5)
        plt.title('Pregled glavnih metrika po prezimenu - validacija')
        plt.tight_layout()
        plt.show()
    else:
        print('Nema rezultata za grafički prikaz.')
