import os
import re
from pathlib import Path
import csv
from datetime import datetime


def convert_to_csv(base_path=None):
    """
    FINALNI UNIVERZALNI konverter - radi sa SVIM formatima:
    - Format 1: timestamp | [x,y,z,fx,fy,fz]
    - Format 2: timestamp | [x,y,z]\n[fx,fy,fz]
    - Format 3: [timestamp, x, y, z, fx, fy, fz]  ← NOVO za akrap_gugo
    """

    if base_path is None:
        base_path = Path(__file__).resolve().parents[2]
    else:
        base_path = Path(base_path)

    print("=" * 80)
    print("FINALNI UNIVERZALNI CSV KONVERTER")
    print("=" * 80)
    print(f"Base path: {base_path}")
    print()

    dataset_dirs = [
        base_path / 'datasets' / 'cuk_pongracic' / 'cuk-brzo_processed',
        base_path / 'datasets' / 'cuk_pongracic' / 'cuk-sporo_processed',
        base_path / 'datasets' / 'cuk_pongracic' / 'pongracic-brzo_processed',
        base_path / 'datasets' / 'cuk_pongracic' / 'pongracic-sporo_processed',
        base_path / 'datasets' / 'norsic_hartl' / 'norsic-brzo_processed',
        base_path / 'datasets' / 'norsic_hartl' / 'norsic_sopro_processed',
        base_path / 'datasets' / 'akrap_gugo' / 'akrap-brzo',
        base_path / 'datasets' / 'akrap_gugo' / 'akrap-sporo',
        base_path / 'datasets' / 'akrap_gugo' / 'gugo-brzo',
        base_path / 'datasets' / 'akrap_gugo' / 'gugo-sporo',
    ]

    total_files = 0
    successful_files = 0

    for dir_path in dataset_dirs:
        if not dir_path.exists():
            continue

        print(f"📁 {dir_path.name}")
        files = sorted([p for p in dir_path.iterdir() if p.is_file() and not p.name.endswith('.csv')])

        if not files:
            print(f"   Nema datoteka")
            continue

        for input_file in files:
            total_files += 1
            # Spremi CSV koristeći stem (ukloni originalnu .txt ekstenziju)
            csv_file = input_file.parent / f"{input_file.stem}.csv"

            result = convert_any_format(str(input_file), str(csv_file))
            if result['success']:
                successful_files += 1
                print(f"   ✅ {input_file.name} → {result['rows']} redaka")
            else:
                print(f"   ❌ {input_file.name}: {result['error']}")

        print()

    print("=" * 80)
    print(f"ZAVRŠENO: {successful_files}/{total_files} uspješno")
    print("=" * 80)


def convert_any_format(input_path, output_path):
    """
    Univerzalna konverzija - automatski detektira format
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return {'success': False, 'rows': 0, 'error': f'Greška čitanja: {e}'}

    if len(lines) == 0:
        return {'success': False, 'rows': 0, 'error': 'Prazna datoteka'}

    # Detektiraj format na osnovu prvog retka
    first_line = lines[0]

    # FORMAT 3: [timestamp, x, y, z, fx, fy, fz] - NEMA separatora |
    if first_line.startswith('[') and '|' not in first_line:
        return convert_format3(lines, output_path)

    # FORMAT 1 ili 2: ima separator |
    elif '|' in first_line:
        return convert_format1_or_2(lines, output_path)

    else:
        return {'success': False, 'rows': 0, 'error': 'Nepoznat format'}


def convert_format3(lines, output_path):
    """
    Format 3: [timestamp, x, y, z, fx, fy, fz]
    Timestamp je UNIX epoch u sekundama
    """
    csv_rows = []

    for line in lines:
        try:
            if not line.startswith('['):
                continue

            # Parsiraj sve vrijednosti
            values = [v.strip() for v in line.strip('[]').split(',')]

            if len(values) != 7:
                continue

            # Prva vrijednost je timestamp (UNIX epoch)
            unix_timestamp = float(values[0])

            # Konvertiraj UNIX timestamp u čitljiv format
            dt = datetime.fromtimestamp(unix_timestamp)
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')

            # Ostale vrijednosti: x, y, z, fx, fy, fz
            coords_and_forces = values[1:7]

            # Kreiraj CSV redak
            csv_row = [timestamp_str] + coords_and_forces
            csv_rows.append(csv_row)

        except Exception:
            continue

    if len(csv_rows) == 0:
        return {'success': False, 'rows': 0, 'error': 'Nema validnih podataka (format 3)'}

    # Zapiši CSV
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'x', 'y', 'z', 'fx', 'fy', 'fz'])
            writer.writerows(csv_rows)
        return {'success': True, 'rows': len(csv_rows), 'error': ''}
    except Exception as e:
        return {'success': False, 'rows': 0, 'error': f'Greška pisanja: {e}'}


def convert_format1_or_2(lines, output_path):
    """
    Format 1: timestamp | [x,y,z,fx,fy,fz]
    Format 2: timestamp | [x,y,z]\n[fx,fy,fz]
    """
    csv_rows = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if '|' not in line:
            i += 1
            continue

        parts = line.split('|')
        if len(parts) != 2:
            i += 1
            continue

        timestamp = parts[0].strip()
        coords = parts[1].strip()

        try:
            if not coords.startswith('['):
                i += 1
                continue

            values = [v.strip() for v in coords.strip('[]').split(',')]

            # Format 1: 6 vrijednosti u jednom retku
            if len(values) == 6:
                csv_row = [timestamp] + values
                csv_rows.append(csv_row)
                i += 1

            # Format 2: 3 vrijednosti + sljedeći redak
            elif len(values) == 3 and i + 1 < len(lines):
                next_line = lines[i + 1].strip()

                if next_line.startswith('[') and '|' not in next_line:
                    values2 = [v.strip() for v in next_line.strip('[]').split(',')]
                    if len(values2) == 3:
                        csv_row = [timestamp] + values + values2
                        csv_rows.append(csv_row)
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        except Exception:
            i += 1
            continue

    if len(csv_rows) == 0:
        return {'success': False, 'rows': 0, 'error': 'Nema validnih podataka (format 1/2)'}

    # Zapiši CSV
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'x', 'y', 'z', 'fx', 'fy', 'fz'])
            writer.writerows(csv_rows)
        return {'success': True, 'rows': len(csv_rows), 'error': ''}
    except Exception as e:
        return {'success': False, 'rows': 0, 'error': f'Greška pisanja: {e}'}


if __name__ == "__main__":
    convert_to_csv()