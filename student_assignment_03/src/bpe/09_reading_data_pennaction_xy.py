import scipy.io

# Put do vaše .mat datoteke
file_path = 'data/raw/1137/1347.mat'

# Učitavanje .mat datoteke
mat_data = scipy.io.loadmat(file_path)

# Izdvajanje x i y koordinata za prvi frame -odgovara 'data/raw/1137/000001.jpg' 
x_first_frame = mat_data['x'][0]
y_first_frame = mat_data['y'][0]

# Formatiranje za ispis
x_first_frame_list = x_first_frame.tolist()
y_first_frame_list = y_first_frame.tolist()

# Ispis koordinata za prvi frame
print("X koordinate za prvi frame:", x_first_frame_list)
print("Y koordinate za prvi frame:", y_first_frame_list)