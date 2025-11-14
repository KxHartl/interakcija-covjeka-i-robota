import numpy as np
import pandas as pd

def calculate_velocity_from_xyz(df, time_step=0.01):
    """
    Izračunava brzinu na temelju x, y, z koordinata.
    Vraca Series istih duljina kao broj redova, NaN za prvi podatak.
    Brzina je euklidska udaljenost izmedju mjerenja / time_step.
    time_step: sekunde za svaki interval (default 0.01)
    """
    dx = df['x'].diff()
    dy = df['y'].diff()
    dz = df['z'].diff()
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    velocity = distance / time_step
    return velocity

def calculate_force_magnitude_df(df):
    """
    Izračunava magnitudu sile iz fx, fy, fz komponenti
    Vraca Series
    """
    return np.sqrt(df['fx']**2 + df['fy']**2 + df['fz']**2)
