# Upute za Rješavanje Zadatka

## Zadatak

Napraviti program koji će preko kamere laptopa gledati studenta, klasificirati točku nosa, nacrtati točku nosa i nacrtati pravokutnik.
Ako se nos nalazi unutar pravokutnika, pravokutnik će biti **zelen**.
Ako nos nije unutar pravokutnika, onda će biti **crven**.

## Rješenje

### Korak 1: Preuzimanje i Instalacija

Udšte u direktorij projekta:

```bash
cd student_assignment_02
```

Installirajte potrebne pakete:

```bash
pip install -r src/requirements.txt
```

### Korak 2: Pokretanje Programa

```bash
cd src
python nose_detection.py
```

### Korak 3: Interakcija

1. Dobit ćete prozor sa video streamom iz kamere
2. Program će automatski detektirati vašu poziciju
3. Vidjet ćete:
   - **Pravokutnik** u sredini ekrana koji se koristi za klasifikaciju
   - **Žuti kružić** na poziciji detektiranog nosa
   - **Zeleni pravokutnik** ako je nos unutar zone
   - **Crveni pravokutnik** ako je nos izvan zone
   - **Status poruka** koja govori "INSIDE" ili "OUTSIDE"
   - **Koordinate nosa** (X, Y)
   - **Confidence score** za sigurnost detektiranja

4. Pritisnite **'q'** za izlaz iz programa

## Tehnički Detalji

### Kako Radi Program

#### 1. Inicijalizacija

Program koristi YOLO11 (YOLOv11) model za detekciju poza:

```python
from ultralytics import YOLO
model = YOLO("yolo11s-pose.pt")
```

YOLO11 je modern model za detekciju pozicije ljudskog tijela koji detektira 17 ključnih točaka:
- Glava (nos, oči, uši)
- Trup (ramena, kuka)
- Ekstremiteti (laktovi, zglobovi, glenjobi, gležnjevi)

#### 2. Detektiranje Nosa

Za svaki frame iz kamere, program radi YOLO11 inference:

```python
results = model(frame)
keypoints = sv.KeyPoints.from_ultralytics(results[0])
nose_position = keypoints.xy[0][0]  # Index 0 = Nos
```

#### 3. Klasifikacija

Program provjerava je li nos unutar ili izvan pravokutnika:

```python
def is_point_in_rectangle(point_x, point_y, rect_x, rect_y, rect_w, rect_h):
    left = rect_x - rect_w // 2
    right = rect_x + rect_w // 2
    top = rect_y - rect_h // 2
    bottom = rect_y + rect_h // 2
    return left <= point_x <= right and top <= point_y <= bottom
```

#### 4. Vizualizacija

Program crta na video stream:

```python
# Crta pravokutnik
color = (0, 255, 0) if is_inside else (0, 0, 255)  # Zeleno ili crveno
cv2.rectangle(frame, (left, top), (right, bottom), color, 3)

# Crta nos
cv2.circle(frame, (nose_x, nose_y), 8, (0, 255, 255), -1)  # Žuto
```

## Parametri koji se mogu Prilagoditi

U datoteci `nose_detection.py`, funkcija `main()` sadrži:

```python
detector = NoseDetector(
    model_path="yolo11s-pose.pt",      # Može biti "yolo11n-pose.pt" za bržu detekciju
    rectangle_width=200,                 # Proširina pravokutnika (pikseli)
    rectangle_height=250,                # Visina pravokutnika (pikseli)
    confidence_threshold=0.5             # Prag sigurnosti (0.0-1.0)
)
```

### Podešavanja:

- **Veličina Pravokutnika**: Povecajte/smanjite `rectangle_width` i `rectangle_height`
- **Osjetljivost**: Snjizite `confidence_threshold` za manju osjetljivost, povecajte za vecu
- **Rezolucija Kamere**: Promijenite vrijednosti u `__init__`:
  ```python
  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  ```

## Koristeći YOLO11 pose primjere s vježbi

Program je baziran na sljedećim primjerima:

### `02_YOLO11_pose_extracted_points.py`

Prikazuje kako ekstraktirati ključne točke:

```python
keypoints = sv.KeyPoints.from_ultralytics(results[0])
nose = keypoints.xy[0][0]  # Index 0 je nos
```

### `03_YOLO11_pose_extracted_line.py`

Prikazuje kako crtati linije između ključnih točaka.

### Koristeće Supervision Biblioteke

```python
import supervision as sv

# Konverzija YOLO rezultata u Supervision format
keypoints = sv.KeyPoints.from_ultralytics(results[0])

# Pristup koordinatama
print(keypoints.xy)           # SVE koordinate
print(keypoints.confidence)   # SVE vrijednosti sigurnosti
```

## Razumijevanje OpenCV Funkcija

### cv2.circle - Crtanje Kružića

```python
cv2.circle(
    image,           # Na koju sliku crtamo
    (x, y),          # Sredina kružića
    radius,          # Radijus kružića
    color,           # Boja (BGR format): (0, 0, 255) = crveno
    thickness        # Debljina linije (-1 = popunjen kružić)
)
```

### cv2.rectangle - Crtanje Pravokutnika

```python
cv2.rectangle(
    image,              # Na koju sliku crtamo
    (top_left_x, top_left_y),     # Gornji lijevi kut
    (bottom_right_x, bottom_right_y),  # Donji desni kut
    color,              # Boja
    thickness           # Debljina linije
)
```

## Boje u BGR Formatu

OpenCV koristi BGR (ne RGB) format:

- **Crveno**: `(0, 0, 255)`
- **Zeleno**: `(0, 255, 0)`
- **Plavo**: `(255, 0, 0)`
- **Bijelo**: `(255, 255, 255)`
- **Crno**: `(0, 0, 0)`
- **Žuto**: `(0, 255, 255)`
- **Cyan**: `(255, 255, 0)`

## Rješavanje Problema

### 1. "ModuleNotFoundError: No module named 'ultralytics'"

Installirajte pakete:
```bash
pip install -r requirements.txt
```

### 2. "Error: Cannot access camera"

Provjerite:
- Je li kamera podržana?
- Je li kamera pravilno priključena?
- Pokušajte drugog drive: `cv2.VideoCapture(1)` umjesto `cv2.VideoCapture(0)`

### 3. "Error loading model"

Provjerite:
- Imate li internet vezu (model se preuzima prvi put)
- Imate li dovoljno memorije (model je ~20MB)
- Ili preuzmi model ručno:
  ```bash
  python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"
  ```

### 4. Sporo Detektiranje

Taje:
- Koristi `yolo11n-pose.pt` umjesto `yolo11s-pose.pt` (manji i brži model)
- Snjizi rezoluciju kamere
- Koristi GPU ako je dostupan

## Dalji Razvoj

### Mogu li Proširiti Program?

Da! Možete:

1. **Detektirati druge dijelove tijela** - Koristite druge indekse keypointa
2. **Spremati Rezultate** - Dodajte video output ili fotografiju
3. **Zvukova Povratna Informacija** - Dodajte zvuk kada se nos pomakne
4. **Statistika** - Pratite koliko je vremena nos bio unutar/izvan
5. **Mušnjenja Kalibracija** - Prilagodite pravokutnik prema kožinoj rezoluciji

## Literatura

- [YOLO11 Dokumentacija](https://docs.ultralytics.com/)
- [OpenCV Dokumentacija](https://docs.opencv.org/)
- [Supervision Biblioteka](https://supervision.roboflow.com/)

## Kompletna Struktura Datoteka

```
interakcija-covjeka-i-robota/
├── student_assignment_02/
│   ├── README.md                 # Glavni README
│   ├── UPUTE.md                  # Ove upute
│   └── src/
│       ├── nose_detection.py      # Glavna aplikacija
│       └── requirements.txt        # Potrebni paketi
└── README.md
```

## Zaključak

Program učiniće:
1. Pristup kameri
2. Detektiranje nosa koristeći YOLO11
3. Klasifikaciju nosa prema poziciji
4. Vizualizaciju rezultata sa zelenim ili crvenim pravokutnikom

Svi potrebni kod i upute su dostupni u ovom direktoriju!
