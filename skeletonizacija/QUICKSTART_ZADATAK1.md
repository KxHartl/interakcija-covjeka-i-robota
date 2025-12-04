# Brza uputa - Zadatak 1

## Brzo pokretanje (3 koraka)

### 1. Instaliraj zavisnosti

```bash
cd skeletonizacija
pip install -r requirements.txt
```

### 2. Provjeri strukturu podataka

Podaci se već nalaze u repozitoriju:

```
skeletonizacija/
  ├── data/
  │   └── raw/
  │       └── 1137/
  │           ├── 000001.jpg  ← 31 slika (000001.jpg - 000031.jpg)
  │           ├── 000002.jpg
  │           ├── ...
  │           ├── 000031.jpg
  │           └── 1347.mat    ← Ground truth keypoints
  └── src/
      └── zadatak1_compare_gt_mediapipe.py
```

### 3. Pokreni analizu

```bash
cd src
python zadatak1_compare_gt_mediapipe.py
```

---

## Što skripta radi?

**Automatski odabir 5 slika:**
- Frame 1 (000001.jpg)
- Frame 5 (000005.jpg)
- Frame 10 (000010.jpg)
- Frame 15 (000015.jpg)
- Frame 20 (000020.jpg)

**Za svaku sliku:**
1. Učitava ground truth keypoints iz `1347.mat`
2. Primjenjuje MediaPipe Pose detekciju
3. Mapira 13 PennAction keypoints na 33 MediaPipe keypoints
4. Izračunava Euclidean distance
5. Generira metrike (RMSE, mean/median/std error)
6. Kreira vizualizaciju (plavo = GT, crveno = MediaPipe)

---

## Output

Rezultati se spremaju u `data/results/zadatak1/`:

```
data/results/zadatak1/
  ├── summary_results.csv               ← Tablica s metrikama
  ├── comparison_frame000001.jpg      ← Vizualizacije
  ├── comparison_frame000005.jpg
  ├── comparison_frame000010.jpg
  ├── comparison_frame000015.jpg
  └── comparison_frame000020.jpg
```

---

## Primjer konzolnog outputa

```
============================================================
ZADATAK 1: Usporedba Ground Truth i MediaPipe Keypoints
============================================================

Učitavanje ground truth iz: ../data/raw/1137/1347.mat
Broj odabranih slika: 5

Procesuiranje: ../data/raw/1137/000001.jpg
  Ground truth učitan: (13, 2)
  MediaPipe detektirao: (33, 2)
  RMSE: 18.45 px
  Mean error: 17.32 px
  ✓ Vizualizacija spremljena: ../data/results/zadatak1/comparison_frame000001.jpg

...

============================================================
RESULTATI USPOREDBE
============================================================

Osnovne metrike po slici:
Slika          | RMSE (px) | Mean Error (px) | ...
000001.jpg     |    18.45  |     17.32       | ...
000005.jpg     |    16.78  |     15.89       | ...
000010.jpg     |    19.23  |     18.45       | ...
000015.jpg     |    17.56  |     16.92       | ...
000020.jpg     |    20.12  |     19.34       | ...

------------------------------------------------------------
Prosječne greške po ključnim točkama (svih 5 slika):
------------------------------------------------------------
 0. head                :  12.45 px
 1. left_shoulder       :  10.23 px
 2. right_shoulder      :  11.87 px
 3. left_elbow          :  18.92 px
 4. right_elbow         :  17.65 px
 5. left_wrist          :  25.43 px  ← Najveća greška
 6. right_wrist         :  23.78 px
 7. left_hip            :   9.12 px  ← Najmanja greška
 8. right_hip           :   8.89 px
 9. left_knee           :  14.56 px
10. right_knee          :  15.23 px
11. left_ankle          :  20.34 px
12. right_ankle         :  19.87 px

============================================================
UKUPNA STATISTIKA (svih 5 slika)
============================================================
Prosječni RMSE:       18.43 px
Prosječna Mean Error: 17.58 px
============================================================

✓ Zadatak završen! Rezultati su u direktoriju: ../data/results/zadatak1
```

---

## Prilagođavanje odabira slika

Ako želiš odabrati druge slike, uredi `src/zadatak1_compare_gt_mediapipe.py`, linija ~267:

```python
selected_frames = [
    1,   # 000001.jpg
    5,   # 000005.jpg
    10,  # 000010.jpg
    15,  # 000015.jpg
    20   # 000020.jpg
]
```

Možeš odabrati bilo koji frame od 1 do 31.

---

## Rješavanje češćih problema

### ModuleNotFoundError: No module named 'cv2'

```bash
pip install opencv-python
```

### ModuleNotFoundError: No module named 'mediapipe'

```bash
pip install mediapipe
```

### FileNotFoundError: .mat datoteka ne postoji

Provjeri da si u `src/` direktoriju:

```bash
cd skeletonizacija/src
python zadatak1_compare_gt_mediapipe.py
```

### MediaPipe ne detektira pozu

Smanji confidence threshold u skripti (linija ~29):

```python
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.3  # Smanji sa 0.5 na 0.3
)
```

---

## Dodatna dokumentacija

- **Detaljno**: Vidi `README_ZADATAK1.md`
- **Pomoćne funkcije**: Vidi `src/utils_keypoint_mapping.py`

---

## Kontakt

Za pitanja:
- GitHub Issues: https://github.com/KxHartl/interakcija-covjeka-i-robota/issues
- Autor: Krešimir Hartl
