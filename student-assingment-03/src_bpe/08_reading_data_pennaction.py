import scipy.io

# Put do vaše .mat datoteke
file_path = 'data/raw/1137/1137.mat'

# Učitavanje .mat datoteke
mat_data = scipy.io.loadmat(file_path)

# Ispis sadržaja svakog ključa osim onih koji počinju s '__'
for key in mat_data:
    if not key.startswith('__'):
        print(f"Podaci za ključ '{key}':")
        print(mat_data[key])
        print()  # Prazan red za lakše čitanje