import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n" + "="*60)
print("ML MODEL TRAINING - THUMBS UP vs THUMBS DOWN")
print("="*60)

# Provjeri da li CSV datoteka postoji
csv_path = 'gesture_data.csv'
if not os.path.exists(csv_path):
    print(f"\n❌ GREŠKA: {csv_path} nije pronaden!")
    print("Prvo trebas pokrenuti 01_collect_gesture_data.py")
    exit(1)

# Ucitaj podatke
print(f"\n1. Ucitavanje podataka iz {csv_path}...")
df = pd.read_csv(csv_path)
print(f"   ✓ Ucitano {len(df)} frameova")
print(f"   Stupci: {df.shape[1]}")

# Odvoji features i labels
X = df.iloc[:, :-1].values  # Svi landmarks (x,y koordinate)
y = df['label'].values       # Labela (0=thumbs_up, 1=thumbs_down)

print(f"\n2. Statistika klasa:")
print(f"   - THUMBS_UP (0): {sum(y==0)} primjera")
print(f"   - THUMBS_DOWN (1): {sum(y==1)} primjera")

# Podjeli podatke na train i test
print(f"\n3. Dijeljenje podataka (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train set: {len(X_train)} primjera")
print(f"   ✓ Test set: {len(X_test)} primjera")

# Normalizacija podataka
print(f"\n4. Normalizacija podataka (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   ✓ Normalizacija zavrsena")

# Treniranje KNN modela
print(f"\n5. Treniranje KNN modela (n_neighbors=3)...")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)
print(f"   ✓ Treniranje zavrseno")

# Evaluacija na test setu
print(f"\n6. Evaluacija modela:")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {accuracy:.2%}")
print(f"\n   Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['THUMBS_UP', 'THUMBS_DOWN']
))

print(f"\n   Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Spremi model i scaler
print(f"\n7. Spremi model i scaler...")
model_path = 'gesture_model.pkl'
scaler_path = 'scaler.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"   ✓ Model spremi: {model_path}")

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Scaler spremi: {scaler_path}")

print(f"\n{'='*60}")
print("✓ TRENIRANJE ZAVRSENO!")
print(f"{'='*60}\n")
