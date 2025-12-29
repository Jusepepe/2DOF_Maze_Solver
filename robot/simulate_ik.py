import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ik_from_coords import compute_ik, animate_arm, A1, A2
import simulations

# --- CONSTANTS ---
DEMO_TYPE = "flower"  # Options: "circle", "flower", "spiral"
ANIMATE = True

def main():
    print(f"Running simulation: {DEMO_TYPE}")
    
    if DEMO_TYPE == "circle":
        x, y = simulations.generate_circle_path()
    elif DEMO_TYPE == "flower":
        x, y = simulations.generate_flower_path()
    elif DEMO_TYPE == "spiral":
        x, y = simulations.generate_spiral_path()
    else:
        print(f"Unknown demo type: {DEMO_TYPE}")
        return

    # Compute IK
    q1, q2, r, feasible = compute_ik(x, y, A1, A2)
    n_feas = int(np.count_nonzero(feasible))
    print(f"Points: {len(x)} | Feasible: {n_feas}")

    if ANIMATE:
        animate_arm(q1, q2, A1, A2, full_workspace=True, desired_x=x, desired_y=y)

if __name__ == "__main__":
    main()
