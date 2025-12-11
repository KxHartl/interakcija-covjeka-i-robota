#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zadatak 1: Test Skripta

Brža provjera dostupnosti podataka i osnovne funkcionalnosti.
Preporuljivo pokrenuti prvo prije ostalih skripti.

Autor: Student
Datum: 2025
"""

import numpy as np
import cv2
from pathlib import Path
import sys

def test_data_folder():
    """Testira dostupnost data foldera i datoteka."""
    print("\n" + "="*70)
    print("TEST 1: Dostupnost Data Foldera")
    print("="*70 + "\n")
    
    data_root = Path("./student-assingment-03/data/raw/hpe/24")
    
    if data_root.exists():
        print(f"✓ Folder postoji: {data_root}\n")
        
        # Prebrojaj datoteke
        rgb_files = list(data_root.glob("frame_*_rgb.png"))
        pose_files = list(data_root.glob("frame_*_pose.txt"))
        depth_files = list(data_root.glob("frame_*_depth.bin"))
        
        print(f"RGB slike:     {len(rgb_files)} datoteka")
        print(f"Pose datoteke: {len(pose_files)} datoteka")
        print(f"Depth mape:    {len(depth_files)} datoteka")
        
        if rgb_files:
            # Pronađi range framea
            frame_ids = sorted([int(f.stem.split('_')[1]) for f in rgb_files])
            print(f"\nRaspon framea: {frame_ids[0]:05d} - {frame_ids[-1]:05d}")
            print(f"Ukupno: {len(frame_ids)} framea")
            
            return data_root, True
        else:
            print("\n✗ Nema RGB datoteka u folderu!")
            return data_root, False
    else:
        print(f"✗ Folder NE postoji: {data_root}")
        print("\nPrilagodi putanju u test_task_1.py\n")
        return data_root, False

def test_single_frame(data_root: Path, frame_id: int = 100):
    """Testira učitavanje jednog framea."""
    print("\n" + "="*70)
    print(f"TEST 2: Učitavanje Frame-a {frame_id:05d}")
    print("="*70 + "\n")
    
    rgb_path = data_root / f"frame_{frame_id:05d}_rgb.png"
    pose_path = data_root / f"frame_{frame_id:05d}_pose.txt"
    depth_path = data_root / f"frame_{frame_id:05d}_depth.bin"
    
    # Test RGB
    if rgb_path.exists():
        img = cv2.imread(str(rgb_path))
        if img is not None:
            h, w = img.shape[:2]
            print(f"✓ RGB: {rgb_path.name}")
            print(f"  Dimenzije: {w}x{h}, Kanali: {img.shape[2]}")
        else:
            print(f"✗ RGB: Nije mogu učitati {rgb_path.name}")
    else:
        print(f"✗ RGB: Datoteka ne postoji {rgb_path.name}")
    
    # Test Pose
    if pose_path.exists():
        try:
            data = np.loadtxt(str(pose_path))
            print(f"✓ Pose: {pose_path.name}")
            print(f"  Oblik: {data.shape}")
            if data.shape == (4, 3):
                print(f"  Rotacija: 3x3 matrica")
                print(f"  Translacija: {data[3, :]} mm")
        except Exception as e:
            print(f"✗ Pose: Greška pri učitavanju - {e}")
    else:
        print(f"✗ Pose: Datoteka ne postoji {pose_path.name}")
    
    # Test Depth
    if depth_path.exists():
        try:
            with open(depth_path, 'rb') as f:
                width = int.from_bytes(f.read(4), 'little')
                height = int.from_bytes(f.read(4), 'little')
            print(f"✓ Depth: {depth_path.name}")
            print(f"  Dimenzije: {width}x{height}")
        except Exception as e:
            print(f"✗ Depth: Greška pri učitavanju - {e}")
    else:
        print(f"⚠  Depth: Datoteka ne postoji (opciono)")

def test_mediapipe():
    """Testira MediaPipe instalaciju."""
    print("\n" + "="*70)
    print("TEST 3: MediaPipe Instalacija")
    print("="*70 + "\n")
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe instaliran: v{mp.__version__}")
        
        # Test FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6
        )
        print(f"✓ FaceMesh model učitan")
        face_mesh.close()
        return True
    except ImportError as e:
        print(f"✗ MediaPipe nije instaliran")
        print(f"  Instaliraj: pip install mediapipe")
        return False
    except Exception as e:
        print(f"✗ Greška: {e}")
        return False

def test_opencv():
    """Testira OpenCV instalaciju."""
    print("\n" + "="*70)
    print("TEST 4: OpenCV Instalacija")
    print("="*70 + "\n")
    
    try:
        import cv2
        print(f"✓ OpenCV instaliran: v{cv2.__version__}")
        return True
    except ImportError:
        print(f"✗ OpenCV nije instaliran")
        print(f"  Instaliraj: pip install opencv-python")
        return False

def test_dependencies():
    """Testira sve ovisnosti."""
    print("\n" + "="*70)
    print("TEST 5: Ostale Ovisnosti")
    print("="*70 + "\n")
    
    dependencies = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib (opciono)',
        'seaborn': 'Seaborn (opciono)',
        'PIL': 'Pillow'
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            is_optional = 'opciono' in name
            status = "⚠ " if is_optional else "✗"
            print(f"{status}{name}")
            if not is_optional:
                all_ok = False
    
    return all_ok

def main():
    """Pokrenuvanje svih testova."""
    print("\n" + "#"*70)
    print("# ZADATAK 1 - TEST SKRIPTA")
    print("#"*70)
    
    # Test 1: Data folder
    data_root, data_ok = test_data_folder()
    
    if not data_ok:
        print("\n⚠️  Zaustavujem - data folder nije dostupan.\n")
        return False
    
    # Test 2: Single frame
    test_single_frame(data_root)
    
    # Test 3: MediaPipe
    mp_ok = test_mediapipe()
    
    # Test 4: OpenCV
    cv_ok = test_opencv()
    
    # Test 5: Dependencies
    deps_ok = test_dependencies()
    
    # Sažetak
    print("\n" + "="*70)
    print("SAŽETAK TESTOVA")
    print("="*70 + "\n")
    
    if data_ok and mp_ok and cv_ok and deps_ok:
        print("✓ SVE PROVJERE SU PROŠLE!")
        print("\nMožeš pokrenuti:")
        print("  python student-assingment-03/src/00_task_1_main.py")
        print("  python student-assingment-03/src/01_task_1_visualization.py")
        print("  python student-assingment-03/src/02_task_1_statistics.py")
        print()
        return True
    else:
        print("✗ NEKE PROVJERE NISU PROŠLE")
        print("\nMolim riješiti probleme prije pokretanja glavnih skripti.\n")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
