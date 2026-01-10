# Gesture Recognition Task 3 - ML Classifier for Thumbs Up/Down

Ovo je rješenje za Samostalni Zadatak 3 - Klasifikator (ML) za dvije nove geste: **THUMBS UP** i **THUMBS DOWN**.

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

### 2. **02_train_gesture_model.py** - Treniranje Modela

**Svrha:** Treniranje KNN ML modela na prikupljenim podacima.

**Kako koristiti:**
```bash
python 02_train_gesture_model.py
```

**Proces:**
1. Ucitaj `gesture_data.csv`
2. Podijeli podatke (80% train, 20% test)
3. Normalizacija (StandardScaler)
4. Treniranje KNN modela (k=3)
5. Evaluacija na test setu
6. Spremi model i scaler

**Izlaz:**
- `gesture_model.pkl` - Treniran KNN model
- `scaler.pkl` - StandardScaler za normalizaciju
- Accuracy i evaluation report

**Primjer rezultata:**
```
Accuracy: 95.00%

Classification Report:
              precision    recall  f1-score   support
   THUMBS_UP       0.95      0.95      0.95         3
  THUMBS_DOWN       0.95      0.95      0.95         3
```

---

### 3. **03_gesture_recognition_ml.py** - Real-Time Predikcija

**Svrha:** Korištenje trenanog modela za real-time prepoznavanje gesti.

**Kako koristiti:**
```bash
python 03_gesture_recognition_ml.py
```

**Karakteristike:**
- ✅ Real-time video feed sa kamerom
- ✅ Detekcija ruke sa MediaPipe Hands
- ✅ ML predikcija sa confidence scores
- ✅ Glađenje rezultata (majority voting na 5 frameova)
- ✅ Prikaz landmarks sa bojama
- ✅ Labela sa gestom i pouzdanošću

**Kontrole:**
- **ESC** ili **'q'** - Izlaz

**Izlaz:** Real-time video sa prepoznatim gestama

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
4:  THUMB_TIP
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

## 🎯 Koraci za Uspješno Rješenje

1. **Prikupljanje Podataka** (30-45 min)
   ```bash
   python 01_collect_gesture_data.py
   ```
   - Izvedite THUMBS UP i THUMBS DOWN pred kamerom
   - Prikupite minimalno 15-20 frameova po gestami
   - Izlazna datoteka: `gesture_data.csv`

2. **Treniranje Modela** (5 min)
   ```bash
   python 02_train_gesture_model.py
   ```
   - Automatski se trenira KNN model
   - Prikazuje se accuracy i evaluation
   - Izlazne datoteke: `gesture_model.pkl`, `scaler.pkl`

3. **Testiranje Real-Time** (10-15 min)
   ```bash
   python 03_gesture_recognition_ml.py
   ```
   - Testiraj model sa novim gestama pred kamerom
   - Provjeri da li pravilno prepoznaje obje geste
   - Spremi screenshot kao dokaz

---

## ✅ Checklist Prije Predaje

- [ ] `gesture_data.csv` sa minimalno 30 frameova (15 po gestama)
- [ ] `01_collect_gesture_data.py` - skriptu za prikupljanje
- [ ] `02_train_gesture_model.py` - skriptu za treniranje
- [ ] `03_gesture_recognition_ml.py` - skriptu za real-time
- [ ] Screenshot gdje se vide obje geste prepoznate
- [ ] Accuracy od modela ≥ 80%
- [ ] Kratak opis kako je model treniran (2-3 reda teksta)

---

## 🔧 Technički Detalji

### Korišteni Algoritmi
- **KNN (K-Nearest Neighbors)** - n_neighbors=3
- **StandardScaler** - normalizacija podataka
- **Train/Test Split** - 80/20

### Metreke Evaluacije
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

### Glađenje Rezultata
- Koristi se buffer od 5 frameova
- Majority voting za stabilnu predikciju
- Prikazuje se confidence score

---

## 📝 Napomene

- Za najbolje rezultate, prikupljajte podatke u različitim osvjetljenjima
- Pazite na pozadinu - trebala bi biti čista
- Izvedite geste sa različitim brzinama
- Ako je accuracy nizak (<80%), prikupljajte više podataka

---

**Autor:** Student Assignment 3  
**Datum:** Januar 2026  
**Predmet:** Interakcija Čovjeka i Robota  
