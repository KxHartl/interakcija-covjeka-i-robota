# Zadatak 1: Usporedba Ground Truth i MediaPipe Predikcija

## Pregled

Ovaj skup skripti analizira i uspoređuje **Ground Truth podatke** (pozicija i rotacija glave iz BIWI dataset-a) sa **MediaPipe FaceMesh predikcijama**.

### Dataset
- **HPE-24 (Head Pose Estimation)**
- Dostupni framea: **50, 150, 250, 350, 420, 508**
- Struktura foldera:
  ```
  data/raw/hpe/24/
  ├── frame_00050_rgb.png
  ├── frame_00050_pose.txt
  ├── frame_00050_depth.bin
  ├── frame_00050_depth_mask.png
  ├── ...
  └── frame_00508_rgb.png
  ```

---

## Skripte

### 1. `00_task_1_main.py` - Glavna Analiza

**Funkcionalnost:**
- Učitava RGB slike i Ground Truth pose podatke
- Ekstraktuje MediaPipe FaceMesh landmarks
- Uspoređuje dostupnost i kvalitetu podataka
- Sprema detaljne rezultate u JSON format

**Pokretanje:**
```bash
python src/00_task_1_main.py
```

**Izlaz:**
- `results/task_1_analysis.json` - Detaljni izvještaj za sve framee

---

### 2. `01_task_1_visualization.py` - Vizualizacija

**Funkcionalnost:**
- Crta 3D koordinatne osi (Ground Truth) na slici
- Prikazuje MediaPipe landmarks kao točke
- Crta bounding box oko detektiranog lica
- Dodaje tekstualne informacije (frame ID, status)
- Sprema rezultate kao PNG slike

**Legenda na slici:**
- **Crvena linija (X os)** - Ground Truth rotacija
- **Zelena linija (Y os)** - Ground Truth rotacija
- **Plava linija (Z os)** - Ground Truth rotacija
- **Žuta točka** - Ishodište (glava)
- **Cyan točke** - MediaPipe keypointi (468 točaka)
- **Zeleni pravokutnik** - Bounding box oko lica

**Pokretanje:**
```bash
python src/01_task_1_visualization.py
```

**Izlaz:**
- `results/task_1_visualization_frame_00050.png`
- `results/task_1_visualization_frame_00150.png`
- ... (6 slika ukupno)

---

### 3. `02_task_1_statistics.py` - Statističke Analize

**Funkcionalnost:**
- Učitava Ground Truth pose podatke
- Pretvara rotacijske matrice u Euler kutove
- Izračunava statističke mjere:
  - **Mean, Std, Min, Max** za svaki kut
  - **Mean, Std, Min, Max** za translaciju (x, y, z)
- Crta distribucije kutova (ako je dostupan Matplotlib)

**Pokretanje:**
```bash
python src/02_task_1_statistics.py
```

**Izlaz:**
- `results/task_1_statistics.json` - Statističke mjere
- `results/euler_distributions.png` - Grafici distribucija (ako je Matplotlib dostupan)

---

## Instalacija Ovisnosti

```bash
pip install -r requirements.txt
```

Ali na mimimumu trebate:
```bash
pip install numpy opencv-python mediapipe
```

Za grafike (opcionalno):
```bash
pip install matplotlib
```

---

## Brzi Start

### 1. Postavi putanje (ako trebaju)

Ako koristiš drugačiju strukturu foldera, ažuriraj `data_root` u svakoj skripti:

```python
# U svakoj skripti
data_root = Path("./student-assingment-03/data/raw/hpe/24")  # Prilagodi po potrebi
```

### 2. Pokreni sve tri skripte

```bash
# Analiza
python src/00_task_1_main.py

# Vizualizacija
python src/01_task_1_visualization.py

# Statistika
python src/02_task_1_statistics.py
```

### 3. Provjeri rezultate

```bash
ls results/
```

Trebao bi vidjeti:
- `task_1_analysis.json` - Detaljni izvještaj
- `task_1_visualization_frame_*.png` - 6 slika
- `task_1_statistics.json` - Statističke mjere

---

## Struktura Rezultata

### analysis.json
```json
{
  "timestamp": "2025-12-11T...",
  "total_frames": 6,
  "data_folder": "...",
  "available_frames": [50, 150, 250, 350, 420, 508],
  "results": [
    {
      "frame_id": 50,
      "status": "success",
      "ground_truth": {
        "euler_angles_deg": [pitch, yaw, roll],
        "translation_m": [x, y, z],
        "rotation_matrix": [[...], [...], [...]]
      },
      "mediapipe": {
        "detected": true,
        "landmark_count": 468
      },
      "comparison": {...}
    },
    ...
  ]
}
```

### statistics.json
```json
{
  "total_frames": 6,
  "frame_ids": [50, 150, 250, 350, 420, 508],
  "euler_angles": {
    "pitch": {"mean": X, "std": X, "min": X, "max": X, "range": X},
    "yaw": {...},
    "roll": {...}
  },
  "translation": {
    "x": {"mean": X, "std": X, "min": X, "max": X, "range": X},
    "y": {...},
    "z": {...}
  }
}
```

---

## Ground Truth Format

### Datoteke

```
frame_XXXXX_pose.txt  → Matrica 4×3 (Rotacija + Translacija)
frame_XXXXX_rgb.png   → RGB slika 640×480
frame_XXXXX_depth.bin → Depth mapa (binarna datoteka)
```

### pose.txt Struktura

```
R11 R12 R13
R21 R22 R23
R31 R32 R33
Tx  Ty  Tz
```

Gdje je:
- **R** - 3×3 rotacijska matrica
- **T** - 3D translacija vektor (u mm)

### Konverzija Kutova

Rotacijska matrica se pretvara u **Euler kutove (XYZ redoslijed)**:
```
Pitch (oko X osi) - Nagib gore-dolje
Yaw   (oko Y osi) - Okret lijevo-desno  
Roll  (oko Z osi) - Nagib na stranu
```

---

## Konstante

### Kalibracija Kamere (BIWI)

```python
K = np.array([
    [575.816, 0.0, 320.0],  # f_x, 0, c_x
    [0.0, 575.816, 240.0],  # 0, f_y, c_y
    [0.0, 0.0, 1.0]         # 0, 0, 1
], dtype=np.float32)
```

---

## Rješavanje Problema

### ⚠️ "Putanja ne postoji"

**Rješenje:** Ažuriraj `data_root` u skripti:

```python
import os
# Ako koristiš relativnu putanju
data_root = Path("./data/raw/hpe/24")

# Ili apsolutnu putanju
data_root = Path("/home/korisnik/projekti/interakcija-covjeka-i-robota/student-assingment-03/data/raw/hpe/24")

# Provjeri postoji li
if data_root.exists():
    print(f"✓ Putanja postoji: {data_root}")
else:
    print(f"✗ Putanja NE postoji: {data_root}")
```

### ⚠️ "MediaPipe nije detektiran"

To je normalno - MediaPipe FaceMesh može ne detektovati lice u nekim framama:
- Loša osvjetljenja
- Dio lica nije vidljiv
- Previše male/velike dimenzije lica

### ⚠️ "Matplotlib nije dostupan"

Grafici neće biti generirani, ali analiza će i dalje raditi:

```bash
pip install matplotlib
```

---

## Dodatne Informacije

### MediaPipe FaceMesh
- 468 3D landmarks na licu
- Relativne 3D koordinate (normalizirane)
- Normalno korišteno za head pose estimation

### BIWI Dataset
- Head Pose Estimation (HPE) dataset
- 3D ground truth kutovi i pozicija
- Korišteno za evaluaciju head pose algoritama

---

## Kontakt i Pomoć

Ako imaš pitanja ili probleme:
1. Provjeri da su sve datoteke na mjestu
2. Provjeri verzije biblioteka: `pip list`
3. Pokreni skripte sa verbose izlazom (ili dodaj `print` iskaze)

---

**Zadnja ažurenja:** 2025-12-11  
**Student Assignment 03** - Zadatak 1
