import numpy as np

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)]

COLOR_MAP = {
    0: "#0000ff",  # Normal
    1: "#ff0000",  # Spalling
    2: "#ff8000",  # Blockage
    3: "#ffff00",  # Corrosion
    4: "#00c800",  # Misalign
    5: "#a020f0",  # Deposit
    6: "#00ffff",  # Displace
    7: "#ff00ff"   # RubberRing
}

def apply_color_mapping(pred_path):
    data = np.loadtxt(pred_path)
    colors = np.array([hex_to_rgb(COLOR_MAP[int(label)]) for label in data[:,3]])
    return np.hstack((data[:,0:3], colors))