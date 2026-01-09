import cv2
import os

# Učitavanje slike
putanja = 'data/raw/1137/000001.jpg'  # Promijeni ovo u put do tvoje slike
slika = cv2.imread(putanja)

cv2.imshow('Slika', slika)

cv2.waitKey(0)
cv2.destroyAllWindows()
