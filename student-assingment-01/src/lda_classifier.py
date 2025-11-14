import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
import seaborn as sns

TRAIN_PREZIMENA = ['norsic', 'hartl']

if __name__ == "__main__":
    df = pd.read_csv('datasets/normalized/all_measurements.csv')
    df_train = df[df['prezime'].isin(TRAIN_PREZIMENA)]
    df_val = df[~df['prezime'].isin(TRAIN_PREZIMENA)]
    X_train, y_train = df_train[['brzina', 'sila']].values, df_train['klasa'].values
    X_val, y_val, prezime_val = df_val[['brzina', 'sila']].values, df_val['klasa'].values, df_val['prezime'].values

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_val)

    # --- Scatter SVI podatci + granica odluke ---
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
    # Prikaži granicu odluke
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

    # --- Ostatak: barplot, heatmap kao prije (preskacem zbog duzine ovdje) ---
