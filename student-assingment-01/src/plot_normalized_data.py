import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv('datasets/normalized/all_measurements.csv')
    print(f"Ukupno mjerenja: {len(df)}")
    print(df['prezime'].value_counts())

    # Scatter plot po prezimenu
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='brzina', y='sila', hue='klasa', style='prezime', data=df, palette={0:'blue', 1:'red'})
    plt.xlabel('Prosječna brzina')
    plt.ylabel('Prosječna sila')
    plt.title('Sav dataset: brzina vs sila, po klasi i prezimenu')
    plt.legend(title="Klasa", labels=["Sporo","Brzo"])
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