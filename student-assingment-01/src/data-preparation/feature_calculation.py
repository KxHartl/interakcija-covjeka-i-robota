import numpy as np
import pandas as pd

def calculate_velocity_from_xyz(df):
    """
    Izračunava brzinu na temelju x, y, z koordinata i vremenskih oznaka.
    Izračun brzine se bazira na stvarnom intervalu dt između dvije uzastopne vrijednosti timestamp-a.
    Za timestamp se očekuje string koji pandas može parsirati u datetime.
    """
    if 'timestamp' not in df.columns:
        raise ValueError("Za izračun brzine potreban je stupac 'timestamp'")
    timestamps = pd.to_datetime(df['timestamp'])
    dt = timestamps.diff().dt.total_seconds()    # dt u sekundama
    dx = df['x'].diff()
    dy = df['y'].diff()
    dz = df['z'].diff()
    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    velocity = distance / dt
    velocity.iloc[0] = np.nan   # Prvi podatak nema prethodnika za diff, pa nan
    return velocity

def calculate_force_magnitude_df(df):
    """
    Izračunava magnitudu sile iz fx, fy, fz komponenti
    Vraca Series
    """
    return np.sqrt(df['fx']**2 + df['fy']**2 + df['fz']**2)
