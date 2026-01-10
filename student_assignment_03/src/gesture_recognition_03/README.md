# Gesture Recognition Task 3 - ML Classifier za Thumbs Up/Down

Ovo je rješenje za Samostalni Zadatak 3 - Klasifikator (ML) za dvije nove geste: **THUMBS UP** i **THUMBS DOWN** korištenjem **SVM (Support Vector Machine)** klasifikatora.

## 🎯 Zašto SVM umjesto KNN?

| Karakteristika | KNN | SVM |
|---|---|---|
| **Brzina** | Spora (trebaš memoriju svih podataka) | Brza (koristi support vectore) |
| **Točnost na gestama** | Dovoljna (~85-90%) | Odlična (~92-98%) |
| **False Positives** | Više | Manje |
| **Optimalno za** | Mali broj klasa | Gestne podatke (kompleksne granice) |
| **Granice klasa** | Jednostavne | Zakrivljene (RBF kernel) |
| **Skupljanje memorije** | Više | Manje |

**SVM sa RBF kernelom** kreira kompleksnije granice između klasa što je idealno za gestne podatke jer:
- Palac u različitim pozicijama → kompleksna feature space
- RBF kernel se prilagođava nelinearnostima
- Manje false positives u neutralnim pozicijama ruke

---

## Faze Rješenja

### 1. **01_collect_gesture_data.py** - Prikupljanje Podataka

**Svrha:** Prikupljanje landmarks podataka za treniranje modela.

**Kako koristiti:**
```bash
python 01_collect_gesture_data.py
```

**Kontrole:**
- **'u'** - Počni prikupljati THUMBS UP podatke
- **'d'** - Počni prikupljati THUMBS DOWN podatke  
- **'n'** - Zaustavi prikupljanje (sljedeća gesta)
- **'q'** - Spremi i zatvori

**Izlaz:** `gesture_data.csv` datoteka sa:
- 42 stupca (21 landmark × 2 koordinate: x, y)
- 1 labela stupac (0=THUMBS_UP, 1=THUMBS_DOWN)
- Minimalno 30 frameova (15 po gestama)

---

### 2. **02_train_gesture_model.py** - Treniranje SVM Modela

**Svrha:** Treniranje SVM ML modela na prikupljenim podacima.

**Kako koristiti:**
```bash
python 02_train_gesture_model.py
```

**Proces:**
1. Ucitaj `gesture_data.csv`
2. Podijeli podatke (80% train, 20% test)
3. Normalizacija (StandardScaler) - OBAVEZNA za SVM!
4. Treniranje SVM modela (kernel='rbf', C=1.0, probability=True)
5. Evaluacija na test setu
6. Spremi model i scaler

**SVM Parametri:**
```python
SVC(kernel='rbf',           # RBF (Radial Basis Function) kernel
    C=1.0,                  # Regularizacijski parametar
    gamma='scale',          # Gamma parametar za RBF
    probability=True,       # Omogući predict_proba() - VAŽNO!
    random_state=42)
```

**Izlaz:**
- `gesture_model.pkl` - Treniran SVM model
- `scaler.pkl` - StandardScaler za normalizaciju
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix

**Primjer rezultata:**
```
Model tip: SVM (Support Vector Machine)
Kernel: RBF (Radial Basis Function)
Test Accuracy: 95.00%

Classification Report:
              precision    recall  f1-score   support
   THUMBS_UP       0.95      0.95      0.95         3
  THUMBS_DOWN       0.95      0.95      0.95         3
```

---

### 3. **03_gesture_recognition_ml.py** - Real-Time Predikcija sa SVM

**Svrha:** Korištenje trenanog SVM modela za real-time prepoznavanje gesti.

**Kako koristiti:**
```bash
python 03_gesture_recognition_ml.py
```

**Karakteristike:**
- ✅ Real-time video feed sa kamerom
- ✅ Detekcija ruke sa MediaPipe Hands
- ✅ **SVM predikcija** sa probability scores
- ✅ Validacija geste (provjera oblika ruke)
- ✅ Glađenje rezultata (majority voting na 8 frameova)
- ✅ Prikaz landmarks sa bojama
- ✅ Labela sa gestom i pouzdanošću
- ✅ **Sprječavanje false positives**

**Validacijski Pragovi:**
```python
MIN_CONFIDENCE_THRESHOLD = 0.72    # Min 72% sigurnosti
MIN_CLASS_PROBABILITY_GAP = 0.15   # Razlika >15% između klasa
CONFIDENCE_BUFFER_SIZE = 8         # 8 frameova prije prikaza
```

**Kontrole:**
- **ESC** ili **'q'** - Izlaz

**Izlaz:** Real-time video sa prepoznatim gestama

**Status Prikaz:**
| Status | Opis |
|--------|------|
| 🟢 **Zeleni landmarks + "THUMBS UP"** | Validna THUMBS UP gesta |
| 🔴 **Crveni landmarks + "THUMBS DOWN"** | Validna THUMBS DOWN gesta |
| ⚫ **Sivi landmarks + "Waiting..."** | Ruka vidljiva ali NIJE u pravoj poziciji |
| ⚪ **Bez ničega** | Nema ruke |

---

## 📊 Struktura Podataka

### CSV Format (gesture_data.csv)
```
lm_0_x,lm_0_y,lm_1_x,lm_1_y,...,lm_20_x,lm_20_y,label
0.5,0.3,0.6,0.2,...,0.55,0.35,0
0.51,0.31,0.61,0.21,...,0.56,0.36,0
...
```

**Legenda:**
- `lm_i_x, lm_i_y` - X i Y koordinate i-tog MediaPipe landmark-a
- `label` - 0 (THUMBS_UP) ili 1 (THUMBS_DOWN)

### MediaPipe Hand Landmarks (21 točka)
```
0:  WRIST
1:  THUMB_CMC
2:  THUMB_MCP
3:  THUMB_IP
4:  THUMB_TIP ⭐ PALAC - VAŽAN ZA GESTE
5:  INDEX_MCP
6:  INDEX_PIP
7:  INDEX_DIP
8:  INDEX_TIP
9:  MIDDLE_MCP
10: MIDDLE_PIP
11: MIDDLE_DIP
12: MIDDLE_TIP
13: RING_MCP
14: RING_PIP
15: RING_DIP
16: RING_TIP
17: PINKY_MCP
18: PINKY_PIP
19: PINKY_DIP
20: PINKY_TIP
```

---

## 🚀 Koraci za Uspješno Rješenje

### Korak 1: Prikupljanje Podataka (~45 min)
```bash
python 01_collect_gesture_data.py
```
- Izvedite THUMBS UP pred kamerom (Palac gore, prstiji zatvoren)
- Izvedite THUMBS DOWN pred kamerom (Palac dolje, prstiji zatvoren)
- Prikupite minimalno 15-20 frameova po gestami
- Izlazna datoteka: `gesture_data.csv`

### Korak 2: Treniranje SVM Modela (~5 min)
```bash
python 02_train_gesture_model.py
```
- Automatski se trenira SVM model sa RBF kernelom
- Prikazuje se accuracy, precision, recall, confusion matrix
- Izlazne datoteke: `gesture_model.pkl`, `scaler.pkl`
- **Provjeri da li je accuracy > 85%**

### Korak 3: Testiranje Real-Time (~15 min)
```bash
python 03_gesture_recognition_ml.py
```
- Testiraj model sa novim gestama pred kamerom
- Provjeri da li pravilno prepoznaje obje geste
- Primjeti sivi prikaz kada nije u pravoj poziciji
- Primjeti zelene/crvene landmarks samo za validne geste
- Spremi screenshot kao dokaz

---

## ✅ Checklist Prije Predaje

- [ ] `gesture_data.csv` sa minimalno 30 frameova (15 po gestama)
- [ ] `01_collect_gesture_data.py` - skriptu za prikupljanje
- [ ] `02_train_gesture_model.py` - skriptu sa **SVM modelom**
- [ ] `03_gesture_recognition_ml.py` - skriptu za real-time sa **validacijom**
- [ ] Screenshot gdje se vide obje geste prepoznate
- [ ] Accuracy od modela ≥ 85%
- [ ] Bez lažnih alarma (false positives) - test neutrlane ruke!
- [ ] Kratak opis:
  - Zašto SVM umjesto KNN?
  - Kako validacija sprječava false positives?
  - Što su confidence threshold i probability gap?

---

## 🔧 Tehnički Detalji

### ML Algoritam: SVM sa RBF Kernelom

**Support Vector Machine (SVM):**
- Pronalazi optimalnu hiperravninu koja razdvaja klase
- Maksimalizira margin između klasa
- Koristi se za binarnu klasifikaciju (THUMBS_UP vs THUMBS_DOWN)

**RBF (Radial Basis Function) Kernel:**
- Preslikava podatke u više-dimenzionalni prostor
- Omogućuje nelinearne granice između klasa
- Idealno za kompleksne gestne podatke

**Normalizacija:**
- OBAVEZNA za SVM!
- StandardScaler pretvara sve features u istu skalu
- Bez normalizacije, SVM ne bi radio pravilno

### Validacija Geste

**3 Provjere Oblika Ruke:**
1. **Palac je vidljiv** - distanca palca od zapešća > 0.05
2. **Ostali prstiji su grupirani** - bounding box < 0.25 × 0.3
3. **Palac je odvojen od prstiju** - distanca palca od indeksa > 0.08

**Kvaliteta Pouzdanosti:**
1. Model je 72%+ siguran
2. Razlika između top 2 klase > 15%
3. 8 uzastopnih frameova potvrdi istu gestu

### Metreke Evaluacije
- **Accuracy** - % točnih predikcija
- **Precision** - % pravilno prepoznatih THUMBS UP/DOWN
- **Recall** - % od svih THUMBS UP/DOWN koje je model prepoznao
- **F1-score** - harmonijska sredina precision i recall
- **Confusion Matrix** - prikaz true/false positives/negatives

---

## 📝 Dodatne Napomene

### Za Najbolje Rezultate:
- Prikupljajte podatke u različitim osvjetljenjima
- Pazite na pozadinu - trebala bi biti čista
- Izvedite geste sa različitim brzinama
- Prikupljajte podatke sa različitih kutova
- Ako je accuracy nizak (<85%), prikupljajte više podataka

### Ako Trebate Podesiti Pragove:
```python
# Niži pragovi = Više tolerancije (više false positives)
MIN_CONFIDENCE_THRESHOLD = 0.68     # Default: 0.72
MIN_CLASS_PROBABILITY_GAP = 0.12    # Default: 0.15

# Viši pragovi = Stroža validacija (manje false positives)
MIN_CONFIDENCE_THRESHOLD = 0.75     # Default: 0.72
MIN_CLASS_PROBABILITY_GAP = 0.18    # Default: 0.15
```

---

## 📚 Reference

- **SVM:** https://scikit-learn.org/stable/modules/svm.html
- **RBF Kernel:** https://en.wikipedia.org/wiki/Radial_basis_function_kernel
- **MediaPipe Hands:** https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- **StandardScaler:** https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

---

**Autor:** Student Assignment 3  
**Datum:** Januar 2026  
**Predmet:** Interakcija Čovjeka i Robota  
**Model:** SVM (Support Vector Machine) sa RBF kernelom  
