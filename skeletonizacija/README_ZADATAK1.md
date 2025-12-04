# Zadatak 1: Usporedba Ground Truth i MediaPipe Keypoints

## Opis zadatka

Usporedba ključnih točaka (keypoints) dobivenih iz PennAction ground truth dataseta s onima detektiranim pomoću MediaPipe Pose biblioteke.

### Zahtjevi zadatka:
1. Odabrati 5 fotografija iz mape `podaci` (PennAction dataset)
2. Za svaku fotografiju učitati ground truth iz PennAction dataseta
3. Za iste ključne točke dobiti poziciju pomoću MediaPipe Pose
4. Rezultate usporediti i prikazati u tablici

---

## Struktura projekta

```
skeletonizacija/
├── data/
│   ├── raw/
│   │   └── PennAction/          # PennAction dataset
│   │       ├── frames/          # Slike/frameovi
│   │       │   ├── 0001/
│   │       │   ├── 0100/
│   │       │   └── ...
│   │       └── labels/          # Ground truth .mat datoteke
│   │           ├── 0001.mat
│   │           ├── 0100.mat
│   │           └── ...
│   └── results/
│       └── zadatak1/            # Output rezultati
├── src/
│   └── zadatak1_compare_gt_mediapipe.py  # Glavna skripta
├── requirements.txt
└── README_ZADATAK1.md
```

---

## Instalacija i postavljanje okruženja

### 1. Preuzimanje PennAction dataseta

PennAction dataset je dostupan na: [http://dreamdragon.github.io/PennAction/](http://dreamdragon.github.io/PennAction/)

```bash
# Kreiranje direktorija za podatke
mkdir -p data/raw/PennAction
cd data/raw/PennAction

# Preuzimanje dataseta (primjer)
wget http://dreamdragon.github.io/PennAction/Penn_Action.tar.gz
tar -xzf Penn_Action.tar.gz
```

### 2. Instalacija Python biblioteka

```bash
# Kreiranje virtualnog okruženja (preporučeno)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ili
venv\Scripts\activate     # Windows

# Instalacija zavisnosti
pip install -r requirements.txt
```

**Potrebne biblioteke:**
- `opencv-python` - Obrada slika
- `numpy` - Numeričke operacije
- `mediapipe` - Detekcija poze
- `scipy` - Učitavanje .mat datoteka
- `pandas` - Analiza i prikaz rezultata
- `matplotlib`, `seaborn` - Vizualizacije (optional)

---

## Pokretanje skripte

### Osnovno pokretanje

```bash
cd src/
python zadatak1_compare_gt_mediapipe.py
```

### Prilagođavanje odabira slika

U skripti `zadatak1_compare_gt_mediapipe.py`, modificirati listu `selected_images`:

```python
selected_images = [
    ("0001", 50),   # (video_id, frame_idx)
    ("0100", 30),
    ("0200", 40),
    ("0300", 20),
    ("0400", 60)
]
```

---

## Kako skripta radi

### 1. **Učitavanje Ground Truth podataka**

PennAction dataset koristi MATLAB `.mat` format za spremanje anotacija. Svaka `.mat` datoteka sadrži:
- `x` - X koordinate 13 ključnih točaka (shape: 13 x broj_frameova)
- `y` - Y koordinate 13 ključnih točaka (shape: 13 x broj_frameova)

```python
data = sio.loadmat('0001.mat')
x_coords = data['x'][:, frame_idx]
y_coords = data['y'][:, frame_idx]
```

### 2. **Mapiranje ključnih točaka**

PennAction ima **13 keypoints**, MediaPipe Pose ima **33 keypoints**. Potrebno je mapiranje:

| PennAction Index | Naziv | MediaPipe Index |
|------------------|-------|------------------|
| 0 | head | 0 (nose) |
| 1 | left_shoulder | 12 |
| 2 | right_shoulder | 11 |
| 3 | left_elbow | 14 |
| 4 | right_elbow | 13 |
| 5 | left_wrist | 16 |
| 6 | right_wrist | 15 |
| 7 | left_hip | 24 |
| 8 | right_hip | 23 |
| 9 | left_knee | 26 |
| 10 | right_knee | 25 |
| 11 | left_ankle | 28 |
| 12 | right_ankle | 27 |

### 3. **MediaPipe Pose detekcija**

```python
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)

results = pose.process(image_rgb)
```

### 4. **Izračun metrika**

Za svaku ključnu točku izračunava se **Euclidean distance**:

```python
distance = sqrt((x_gt - x_mp)^2 + (y_gt - y_mp)^2)
```

**Agregirane metrike:**
- **RMSE** (Root Mean Square Error)
- **Mean Error** (prosječna greška)
- **Median Error**
- **Standard Deviation**
- **Max/Min Error**

### 5. **Vizualizacija**

Generiraju se slike s:
- **Plavim točkama** - Ground truth keypoints
- **Crvenim točkama** - MediaPipe keypoints

---

## Rezultati

### Output datoteke

Skripta generira sledeće rezultate u `data/results/zadatak1/`:

1. **`summary_results.csv`** - Tablica s metrikama za svaku sliku:
   ```csv
   Slika,RMSE (px),Mean Error (px),Median Error (px),Std Error (px),Max Error (px),Min Error (px)
   000050.jpg,15.32,14.87,13.21,8.45,35.67,2.14
   ...
   ```

2. **`comparison_XXXX_frameYY.jpg`** - Vizualizacije usporedbe za svaku sliku

### Primjer konzolnog outputa

```
============================================================
ZADATAK 1: Usporedba Ground Truth i MediaPipe Keypoints
============================================================

Procesuiranje: ../data/raw/PennAction/frames/0001/000050.jpg
  Ground truth učitan: (13, 2)
  MediaPipe detektirao: (33, 2)
  RMSE: 15.32 px
  Mean error: 14.87 px
  ✓ Vizualizacija spremljena: ../data/results/zadatak1/comparison_0001_frame50.jpg

...

============================================================
RESULTATI USPOREDBE
============================================================

Osnovne metrike po slici:
              Slika  RMSE (px)  Mean Error (px)  Median Error (px)  Std Error (px)  Max Error (px)  Min Error (px)
      000050.jpg      15.32            14.87              13.21            8.45           35.67            2.14
      ...

------------------------------------------------------------
Prosječne greške po ključnim točkama (svih 5 slika):
------------------------------------------------------------
 0. head                :  12.45 px
 1. left_shoulder       :  10.23 px
 2. right_shoulder      :  11.87 px
 3. left_elbow          :  18.92 px
 4. right_elbow         :  17.65 px
 5. left_wrist          :  25.43 px
 6. right_wrist         :  23.78 px
 7. left_hip            :   9.12 px
 8. right_hip           :   8.89 px
 9. left_knee           :  14.56 px
10. right_knee          :  15.23 px
11. left_ankle          :  20.34 px
12. right_ankle         :  19.87 px

============================================================
UKUPNA STATISTIKA (svih 5 slika)
============================================================
Prosječni RMSE:       16.78 px
Prosječna Mean Error: 15.92 px
============================================================
```

---

## Interpretacija rezultata

### Očekivani rangovi grešaka

Prema literaturi[1][2], tipične performanse MediaPipe Pose:

- **Dobro**: RMSE < 15 px
- **Primjereno**: RMSE 15-30 px
- **Loše**: RMSE > 30 px

### Faktori koji utječu na točnost:

1. **Kvaliteta slike** - Rezolucija, oštrina, osvjetljenje
2. **Poze** - Ekstremne poze imaju veću grešku
3. **Okluzija** - Prekrivene ključne točke (npr. ruke iza leđa)
4. **Udaljenost osobe** - Osobe bliže kameri = veća točnost

### Ključne točke s većim greškama

Obično:
- **Zapružja** (wrist) - Najveća greška zbog male veličine i pokretljivosti
- **Glavežnji** (ankle) - Teško je precizno detektirati
- **Laktovi** (elbow) - Umjerena greška u dinamičnim pokretima

Bolja točnost:
- **Ramena** (shoulder) - Stabilnije, veće točke
- **Bokovi** (hip) - Centar mase, lakše za detekciju

---

## Reference

[1] Segu, M. et al. (2020). "Depth-Aware Action Descriptors for Human Activity Recognition". *arXiv:2011.13399*

[2] Bazarevsky, V. et al. (2020). "BlazePose: On-device Real-time Body Pose tracking". *Google AI Blog*

[3] Zhang, W. et al. (2013). "From Actemes to Action: A Strongly-supervised Representation for Detailed Action Understanding". *ICCV 2013* (PennAction dataset paper)

---

## Rješavanje problema

### Problem: `.mat` datoteke se ne mogu učitati

```bash
pip install scipy==1.11.0
```

### Problem: MediaPipe ne detektira osobu

- Provjeriti je li osoba u potpunosti vidljiva na slici
- Povećati kvalitetu slike
- Smanjiti `min_detection_confidence` parametar (npr. na 0.3)

### Problem: Slike se ne nalaze na očekivanoj putanji

Prilagoditi `base_dir` u `main()` funkciji:

```python
base_dir = Path("/putanja/do/vašeg/PennAction/dataseta")
```

---

## Kontakt

Autor: Krešimir Hartl  
Kolegij: Interakcija čovjeka i robota  
Datum: Prosinac 2025
