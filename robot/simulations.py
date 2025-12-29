import numpy as np

def generate_circle_path(center=(-0.07, 0.13), radius=0.04, steps=180):
    rad = 2.0 * np.pi * (np.arange(steps) / float(steps))
    x = center[0] + radius * np.cos(rad)
    y = center[1] + radius * np.sin(rad)
    return x, y

def generate_flower_path(center=(-0.06, 0.10), radius=0.04, steps=180):
    rad = 2.0 * np.pi * (np.arange(steps) / float(steps))
    # r = radius * (1 + sin(5*theta))
    # x = cx + r * cos(theta)
    # y = cy + r * sin(theta)
    # The original code logic was: 
    # demo_x = -0.06 + 0.04 * (1 + np.sin(5 * rad) * np.cos(rad))
    # Close but let's match the original exactly:
    x = center[0] + radius * (1 + np.sin(5 * rad) * np.cos(rad))
    # demo_y =  0.10 + 0.04 * (1 + np.sin(5 * rad) * np.sin(rad))
    y = center[1] + radius * (1 + np.sin(5 * rad) * np.sin(rad))
    return x, y

def generate_spiral_path(center=(0, 0), max_radius=0.1, coils=3, steps=200):
    # Just in case they want a spiral
    theta = np.linspace(0, coils * 2 * np.pi, steps)
    r = np.linspace(0, max_radius, steps)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return x, y
