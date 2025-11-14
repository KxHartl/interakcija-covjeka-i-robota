import os
import re
from pathlib import Path

def process_norsic_files(base_path=None):
    """
    Obrađuje sve Norsic dataset datoteke i spaja dva retka u jedan.

    Struktura prije:
    timestamp | [x, y, z]
    timestamp | [fx, fy, fz]

    Struktura poslije:
    timestamp | [x, y, z, fx, fy, fz]
    """

    # Ako nije eksplicitno zadano, postavi `base_path` na direktorij projekta
    # koji sadrži folder `datasets`. Ovo omogućava pokretanje skripte iz bilo kojeg
    # radnog direktorija, a putanje će uvijek biti relativne prema ovom skriptnom
    # modulu (tj. repozitoriju/student-assingment-01).
    if base_path is None:
        # Skripta se nalazi u: <project>/src/data-preparation/
        # Krenemo 2 nivoa gore da dođemo do `src` roditelja, pa treći da dođemo do root
        base_path = Path(__file__).resolve().parents[2]
    else:
        base_path = Path(base_path)

    # Direktoriji za obradu (relativno prema base_path)
    norsic_dirs = [
        base_path / 'datasets' / 'norsic_hartl' / 'norsic-brzo',
        base_path / 'datasets' / 'norsic_hartl' / 'norsic_sopro'
    ]

    for dir_path in norsic_dirs:
        full_path = Path(dir_path)
        print(full_path)
        if not full_path.exists():
            print(f"Upozorenje: Direktorij {full_path} ne postoji!")
            continue

        # Kreiraj novi direktorij za obrađene datoteke
        processed_dir = full_path.parent / (full_path.name + '_processed')
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Obradi sve datoteke u direktoriju
        files = sorted([p.name for p in full_path.iterdir() if p.is_file()])

        for filename in files:
            input_file = full_path / filename
            output_file = processed_dir / filename

            process_single_file(str(input_file), str(output_file))
            print(f"Obrađena datoteka: {filename}")

    print("\nObrada završena!")

def process_single_file(input_path, output_path):
    """
    Obrađuje pojedinačnu datoteku i spaja parove redaka.
    """
    with open(input_path, 'r') as f_in:
        lines = f_in.readlines()

    processed_lines = []
    i = 0

    while i < len(lines) - 1:
        # Uzmi dva uzastopna retka
        line1 = lines[i].strip()
        line2 = lines[i+1].strip()

        # Parsiraj prvi redak (timestamp | [x, y, z])
        parts1 = line1.split('|')
        if len(parts1) != 2:
            i += 1
            continue

        timestamp1 = parts1[0].strip()
        coords1 = parts1[1].strip()

        # Parsiraj drugi redak (timestamp | [fx, fy, fz])
        parts2 = line2.split('|')
        if len(parts2) != 2:
            i += 1
            continue

        timestamp2 = parts2[0].strip()
        coords2 = parts2[1].strip()

        # Provjeri jesu li timestampovi približno jednaki (unutar 1ms razlike)
        # Koristi prvi timestamp

        # Izvuci vrijednosti iz zagrada
        # coords1 format: [x, y, z]
        # coords2 format: [fx, fy, fz]

        # Ukloni zagrade i podijeli po zarezima
        values1 = coords1.strip('[]').split(',')
        values2 = coords2.strip('[]').split(',')

        if len(values1) == 3 and len(values2) == 3:
            # Spoji sve vrijednosti
            combined_values = values1 + values2
            combined_coords = '[' + ', '.join([v.strip() for v in combined_values]) + ']'

            # Kreiraj novi redak
            new_line = f"{timestamp1} | {combined_coords}\n"
            processed_lines.append(new_line)

        # Prijeđi na sljedeći par redaka
        i += 2

    # Zapiši obrađene retke u novu datoteku
    with open(output_path, 'w') as f_out:
        f_out.writelines(processed_lines)

if __name__ == "__main__":
    # Po defaultu skripta računa base path iz mjesta datoteke, tako da se
    # može pokretati iz bilo kojeg direktorija.
    process_norsic_files()
