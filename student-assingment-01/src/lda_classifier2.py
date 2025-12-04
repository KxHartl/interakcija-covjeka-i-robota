import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
import seaborn as sns

TRAIN_PREZIMENA = ['norsic', 'hartl']  # <-- Postavi koje koristiš za treniranje

if __name__ == "__main__":
    df = pd.read_csv('datasets/normalized/all_measurements.csv')
    df_train = df[df['prezime'].isin(TRAIN_PREZIMENA)]
    df_val = df[~df['prezime'].isin(TRAIN_PREZIMENA)]
    X_train, y_train = df_train[['brzina', 'sila']].values, df_train['klasa'].values
    X_val, y_val, prezime_val = df_val[['brzina', 'sila']].values, df_val['klasa'].values, df_val['prezime'].values

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_val)

    # --- Scatter plot svih podataka + granica odluke ---
    plt.figure(figsize=(8,6))
    prezimena = df['prezime'].unique()
    colors = sns.color_palette('tab10', n_colors=len(prezimena))
    marker_dict = {0:'o', 1:'X'}
    for i, prezime in enumerate(prezimena):
        for klasa in [0,1]:
            subset = df[(df['prezime']==prezime) & (df['klasa']==klasa)]
            if not subset.empty:
                plt.scatter(subset['brzina'], subset['sila'], c=[colors[i]], marker=marker_dict[klasa], s=70,
                            label=f"{prezime} {'sporo' if klasa==0 else 'brzo'}")
    x_min, x_max = df['brzina'].min()-.01, df['brzina'].max()+.01
    y_min, y_max = df['sila'].min()-5, df['sila'].max()+5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=1, alpha=0.15, colors=['blue','red'])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Legenda")
    plt.xlabel('Prosječna brzina')
    plt.ylabel('Prosječna sila')
    plt.title('Sav dataset s LDA granicom odluke')
    plt.tight_layout()
    plt.show()

    # --- Rezultati po prezimenu ---
    results_summary = []
    prezimena_val = sorted(set(prezime_val))
    for prezime in prezimena_val:
        idx = (prezime_val == prezime)
        if idx.sum()==0:
            continue
        y_val_p = y_val[idx]
        y_pred_p = y_pred[idx]
        acc = accuracy_score(y_val_p, y_pred_p)
        prec = precision_score(y_val_p, y_pred_p, labels=[0,1], average=None, zero_division=0)
        rec = recall_score(y_val_p, y_pred_p, labels=[0,1], average=None, zero_division=0)
        f1s = f1_score(y_val_p, y_pred_p, labels=[0,1], average=None, zero_division=0)
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
    # --- Barplot za accuracy po prezimenu ---
    if not res_df.empty:
        plt.figure(figsize=(9,6))
        plt.bar(res_df['prezime'], res_df['accuracy'], color='royalblue')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.xlabel('Prezime (validacija)')
        plt.title(f'Accuracy na svako prezime (Test na {TRAIN_PREZIMENA})')
        plt.tight_layout()
        plt.show()
        # --- Heatmap za ostale metrike ---
        res_hm = res_df.set_index('prezime')[['precision_sporo','precision_brzo','recall_sporo','recall_brzo','f1_sporo','f1_brzo','mcc']]
        plt.figure(figsize=(12,6))
        sns.heatmap(res_hm, annot=True, fmt='.2f', cmap='YlGnBu', center=0.5)
        plt.title('Pregled glavnih metrika po prezimenu - validacija')
        plt.tight_layout()
        plt.show()
    else:
        print('Nema rezultata za grafički prikaz.')
