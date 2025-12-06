# Student Assignment 02 - Nose Detection and Classification

## Opis Zadatka

Program koji detektira točku nosa studenta koristeći kameru laptopa i klasificira je prema poziciji:
- **Zeleni pravokutnik**: Nos je unutar pravokutnika
- **Crveni pravokutnik**: Nos je izvan pravokutnika

## Korištene Tehnologije

- **YOLO11**: YOLOv11 pose detection model za detektiranje točaka ljudskog tijela
- **OpenCV**: Za rad s videom i crtanje na video stream-u
- **Supervision**: Za lakši rad s YOLO detektiranim keypoint-ima
- **Python 3.8+**: Programski jezik

## Instalacija

### 1. Kloniranje Repozitorija

```bash
git clone https://github.com/KxHartl/interakcija-covjeka-i-robota.git
cd interakcija-covjeka-i-robota/student-assignment-02
```

### 2. Instalacija Potrebnih Paketa

```bash
pip install -r src/requirements.txt
```

### 3. Preuzimanje YOLO11 Modela

Model će se automatski preuzeti pri prvom pokretanju, ili ga možete ručno preuzeti:

```bash
cd src
python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"
```

Alternativno, možete kopirati model iz `/luka/HRI/yolo11s-pose.pt` ako je dostupan.

## Pokretanje Programa

```bash
cd src
python nose_detection.py
```

## Kako Program Radi

### Glavni Koraci:

1. **Inicijalizacija**: Program učitava YOLO11 pose detection model i otvara web kameru
2. **Detektiranje**: Za svaki frame iz kamere, YOLO11 model detektira sve ključne točke ljudskog tijela
3. **Ekstraktiranje Nosa**: Iz detektiranih ključnih točaka, izvlači se pozicija nosa (index 0)
4. **Klasifikacija**: Provjerava je li nos unutar ili izvan centralno pozicioniranog pravokutnika
5. **Vizualizacija**: 
   - Crtà pravokutnik (zeleni ili crveni ovisno o poziciji nosa)
   - Označava točku nosa žutim kružićem
   - Prikazuje trenutni status (INSIDE/OUTSIDE)
   - Prikazuje koordinate nosa i confidence score

### Parametri:

- **Rectangle Width**: 200 piksela (proširina pravokutnika)
- **Rectangle Height**: 250 piksela (visina pravokutnika)
- **Confidence Threshold**: 0.5 (minimalna sigurnost detektiranja)
- **Camera Resolution**: 1280x720 piksela

## Korištenje

### Tipke:
- **q** - Zaustavljanje programa i zatvaranje aplikacije

### Ekranski Prikaz:

Program prikazuje:
- Zeleni pravokutnik kada je nos unutar određene zone
- Crveni pravokutnik kada je nos izvan zone
- Žuti kružić na poziciji detektiranog nosa
- Status poruka (INSIDE/OUTSIDE)
- Koordinate nosa (X, Y)
- Confidence score (0.0-1.0)
- Broj obrađenih frejmova

## Struktura Datoteka

```
student-assignment-02/
├── README.md              # Ovaj dokument
├── src/
│   ├── nose_detection.py  # Glavna aplikacija
│   └── requirements.txt    # Potrebni Python paketi
└── examples/              # Primjeri korištenja (opciono)
```

## Технички Detalji

### YOLO11 Keypoint Indeksi:

```
0: Nose (Nos)
1: Left Eye
2: Right Eye
3: Left Ear
4: Right Ear
5: Left Shoulder
6: Right Shoulder
7: Left Elbow
8: Right Elbow
9: Left Wrist
10: Right Wrist
11: Left Hip
12: Right Hip
13: Left Knee
14: Right Knee
15: Left Ankle
16: Right Ankle
```

### Referentne Datoteke:

Program je baziran na primjerima iz `/luka/HRI/` direktorija:
- `02_YOLO11_pose_extracted_points.py` - Ekstraktiranje ključnih točaka
- `03_YOLO11_pose_extracted_line.py` - Crtanje linija između točaka
- `04_YOLO11_pose_depth_estimation.py` - Procjena dubine

## Rješavanje Problema

### Greška: "Error: Cannot access camera"

- Provjerite je li kamera podržana i pravilno priključena
- Provjerite dozvole za pristup kameri (Linux/Mac)
- Pokušajte sa `cv2.VideoCapture(1)` ako ste koristili `cv2.VideoCapture(0)`

### Greška: "Error loading model"

- Provjerite ima li dovoljno memorije
- Provjerite internetu vezu (model se preuzima prvi put)
- Ručno preuzmi model: `python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"`

### Sporo Detektiranje

- Koristite `yolo11n-pose.pt` umjesto `yolo11s-pose.pt` (brži ali manje točan)
- Snizite rezoluciju kamere u kodu
- Koristite GPU ako je dostupan (CUDA)

## Dodatne Opcije

Možete prilagoditi program mijenjanjem parametara u `main()` funkciji:

```python
detector = NoseDetector(
    model_path="yolo11s-pose.pt",     # Putanja do modela
    rectangle_width=200,                # Proširina pravokutnika
    rectangle_height=250,               # Visina pravokutnika
    confidence_threshold=0.5            # Prag sigurnosti
)
```

## Sve Datoteke

Sve datoteke potrebne za pokretanje programa nalaze se u `/src/` direktoriju:
- `nose_detection.py` - Glavna aplikacija
- `requirements.txt` - Potrebni paketi

## Autori

Autor: Student Assignment 02
Tehnologije: YOLO11, OpenCV, Supervision
Datum: 2025

## Licence

Apache 2.0 (kao glavni repozitorij)
