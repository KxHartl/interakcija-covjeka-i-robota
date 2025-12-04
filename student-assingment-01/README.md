# Plotanje originalnih podataka

## Opis
Python program za plotanje originalnih podataka iz snimljenih kretnji robota.

Program učitava CSV datoteke s zapisima pozicije i sile izvršnog člana robota te kreira grafove gdje:
- **X-os**: Vrijeme [s]
- **Y-os lijevo**: Brzina [m/s] - izračunata iz pozicija
- **Y-os desno**: Sila [N] - magnituda sile

## Format ulaznih podataka

CSV datoteke moraju sadržavati sljedeće kolone:
- `x`, `y`, `z` - pozicije izvršnog člana [m]
- `fx`, `fy`, `fz` - komponente sile [N]

Primjer formata:
```csv
x,y,z,fx,fy,fz
0.450,0.234,0.567,2.3,1.5,0.8
0.451,0.235,0.568,2.4,1.6,0.9
...
```

## Struktura direktorija

```
student-assingment-01/
├── datasets/                    # Direktorij s CSV datotekama
│   ├── norsic/
│   │   ├── norsic_brza01.csv
│   │   ├── norsic_spora01.csv
│   │   └── ...
│   └── ...
└── src/
    └── data-preparation/
        ├── plot_original_data.py    # Glavni program
        ├── README.md                # Upute (ova datoteka)
        └── plots/                   # Direktorij s generiranim grafovima (auto-kreiran)
            ├── norsic_brza01_plot.png
            ├── norsic_spora01_plot.png
            └── ...
```

## Instalacija potrebnih paketa

```bash
pip install pandas matplotlib numpy
```

## Korištenje

### 1. Plotanje svih datoteka iz direktorija

Pokrenite program iz direktorija `src/data-preparation/`:

```bash
python plot_original_data.py
```

Program će automatski pronaći sve CSV datoteke u direktoriju `../../datasets` (rekurzivno) i kreirati grafove u direktoriju `plots/`.

### 2. Plotanje jedne datoteke

Za plotanje jedne specifične datoteke, uredite `__main__` blok u programu:

```python
if __name__ == "__main__":
    plot_data('../../datasets/norsic/norsic_brza01.csv')
```

ili unutar drugog Python skripta:

```python
from plot_original_data import plot_data

plot_data('putanja/do/datoteke.csv', output_dir='izlazni_direktorij')
```

## Funkcije

### `calculate_velocity(positions, time_step=0.01)`
Izračunava brzinu izvršnog člana iz pozicija.

**Parametri:**
- `positions`: DataFrame s kolonama x, y, z
- `time_step`: Vremenski korak između mjerenja (default 0.01s)

**Povratna vrijednost:** Niz brzina [m/s]

### `calculate_force_magnitude(forces)`
Izračunava magnitudu sile iz komponenti.

**Parametri:**
- `forces`: DataFrame s kolonama fx, fy, fz

**Povratna vrijednost:** Niz magnitud sila [N]

### `plot_data(csv_file, output_dir='plots')`
Učitava CSV i crta graf.

**Parametri:**
- `csv_file`: Put do CSV datoteke
- `output_dir`: Direktorij za spremanje grafova

### `plot_all_data(data_dir='../../datasets', output_dir='plots')`
Plotanje svih CSV datoteka iz direktorija.

**Parametri:**
- `data_dir`: Direktorij s CSV datotekama
- `output_dir`: Direktorij za spremanje grafova

## Napomene

- Program pretpostavlja da su mjerenja snimljena svakih 0.01 sekundi (100 Hz)
- Ako je frekvencija mjerenja drugačija, promijenite parametar `time_step` u funkciji `calculate_velocity()`
- Brzina se izračunava kao euklidska udaljenost između uzastopnih pozicija podijeljena s vremenskim korakom
- Sila se prikazuje kao magnituda (norma) vektora sile
- Program rekurzivno pretražuje sve poddirektorije u `datasets/` direktoriju

## Zadatak

Program je izrađen u sklopu predmeta **Interakcija čovjeka i robota** za Zadatak 1: Klasifikacija brzih i sporih kretnji robota.

Dataset za treniranje: Norsic  
Ostali dataseti: Za validaciju
