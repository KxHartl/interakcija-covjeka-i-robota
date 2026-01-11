import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n" + "="*60)
print("ML MODEL TRAINING - SVM CLASSIFIER")
print("THUMBS UP vs THUMBS DOWN")
print("="*60)

# Provjeri da li CSV datoteka postoji
csv_path = './outputs/gesture_data.csv'
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

# Treniranje SVM modela
print(f"\n5. Treniranje SVM modela (RBF kernel)...")
print(f"   Parametri: kernel='rbf', C=1.0, gamma='scale', probability=True")
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model.fit(X_train_scaled, y_train)
print(f"   ✓ Treniranje zavrseno")
print(f"   Broj support vectors: {len(model.support_vectors_)}")

# Evaluacija na train setu
print(f"\n6. Evaluacija na TRAIN setu:")
y_train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"   Train Accuracy: {train_accuracy:.2%}")

# Evaluacija na test setu
print(f"\n7. Evaluacija na TEST setu:")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Test Accuracy: {accuracy:.2%}")
print(f"\n   Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['THUMBS_UP', 'THUMBS_DOWN']
))

print(f"\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"   True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Spremi model i scaler
print(f"\n8. Spremi model i scaler...")
model_path = './outputs/gesture_model.pkl'
scaler_path = './outputs/scaler.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"   ✓ Model spremi: {model_path}")

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Scaler spremi: {scaler_path}")

print(f"\n{'='*60}")
print("✓ TRENIRANJE ZAVRSENO!")
print(f"  Model tip: SVM (Support Vector Machine)")
print(f"  Kernel: RBF (Radial Basis Function)")
print(f"  Test Accuracy: {accuracy:.2%}")
print(f"{'='*60}\n")
