import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv('datasets/normalized/all_measurements.csv')
    print(f"Ukupno mjerenja: {len(df)}")
    print(df['prezime'].value_counts())

    # Scatter plot: za svako prezime posebna boja, marker: 'o' (sporo), 'X' (brzo)
    plt.figure(figsize=(8,6))
    prezimena = df['prezime'].unique()
    colors = sns.color_palette('tab10', n_colors=len(prezimena))
    marker_dict = {0:'o', 1:'X'}
    for i, prezime in enumerate(prezimena):
        for klasa in [0,1]:
            subset = df[(df['prezime']==prezime) & (df['klasa']==klasa)]
            if not subset.empty:
                plt.scatter(subset['brzina'], subset['sila'], 
                            c=[colors[i]], marker=marker_dict[klasa], s=70,
                            label=f"{prezime} {'sporo' if klasa==0 else 'brzo'}")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Legenda")
    plt.xlabel('Prosječna brzina')
    plt.ylabel('Prosječna sila')
    plt.title('Sav dataset: brzina vs sila, boja=prezime, marker=klasa')
    plt.tight_layout()
    plt.show()

    # Boxplot za brzinu po prezimenu i klasi
    plt.figure(figsize=(10,6))
    sns.boxplot(x='prezime', y='brzina', hue='klasa', data=df, palette={0:'blue', 1:'red'})
    plt.ylabel('Prosječna brzina')
    plt.title('Distribucija brzine po prezimenima')
    plt.tight_layout()
    plt.show()

    # Boxplot za silu po prezimenu i klasi
    plt.figure(figsize=(10,6))
    sns.boxplot(x='prezime', y='sila', hue='klasa', data=df, palette={0:'blue', 1:'red'})
    plt.ylabel('Prosječna sila')
    plt.title('Distribucija sile po prezimenima')
    plt.tight_layout()
    plt.show()