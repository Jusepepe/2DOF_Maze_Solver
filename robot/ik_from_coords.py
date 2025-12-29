import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- CONSTANTS ---
A1 = 0.0848
A2 = 0.1060
INPUT_FILE = "coords_camera.csv"
ANIMATE = True
SAVE_ANIM = False  # Set to "output.gif" to save
TARGET_WORKSPACE_RATIO = 0.5  # Fit content to 50% of max reach

def compute_ik(x: np.ndarray, y: np.ndarray, a1: float, a2: float):
    r = np.sqrt(x**2 + y**2)
    r_safe = np.where(r == 0, 1e-12, r)

    CosA = (r**2 + a1**2 - a2**2) / (2.0 * r_safe * a1)
    CosC = (a1**2 + a2**2 - r**2) / (2.0 * a1 * a2)

    CosA_clamped = np.clip(CosA, -1.0, 1.0)
    CosC_clamped = np.clip(CosC, -1.0, 1.0)

    SenA = np.sqrt(1.0 - CosA_clamped**2)
    SenC = np.sqrt(1.0 - CosC_clamped**2)

    angleA = np.arctan2(SenA, CosA_clamped)
    angleC = np.arctan2(SenC, CosC_clamped)

    q1 = np.arctan2(y, x) - angleA
    q2 = np.pi - angleC

    feasible = (np.abs(CosA) <= 1.0) & (np.abs(CosC) <= 1.0) & (r > 0)
    return q1, q2, r, feasible

def animate_arm(q1: np.ndarray, q2: np.ndarray, a1: float, a2: float,
                 save_path: str | None = None, full_workspace: bool = True,
                 desired_x: np.ndarray | None = None,
                 desired_y: np.ndarray | None = None,
                 interval_ms: int = 50, hold_start_s: float = 1.0):
    
    hold_frames = max(0, int(round(hold_start_s * 1000.0 / max(1, interval_ms))))
    if hold_frames > 0:
        q1_ext = np.concatenate([np.zeros(hold_frames), q1])
        q2_ext = np.concatenate([np.zeros(hold_frames), q2])
    else:
        q1_ext, q2_ext = q1, q2

    n = len(q1_ext)
    x1 = a1 * np.cos(q1_ext)
    y1 = a1 * np.sin(q1_ext)
    x2 = x1 + a2 * np.cos(q1_ext + q2_ext)
    y2 = y1 + a2 * np.sin(q1_ext + q2_ext)

    L = a1 + a2
    margin = 0.02
    
    if full_workspace:
        x_min, x_max = -L, L
        y_min, y_max = -L, L
    else:
        # Dynamic limits based on path
        x_min = min(-L, float(x2.min()))
        x_max = max(L, float(x2.max()))
        y_min = min(-L, float(y2.min()))
        y_max = max(L, float(y2.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title('2-DOF Arm Simulation')

    link_line, = ax.plot([], [], '-o', lw=3, ms=6)
    path_line, = ax.plot([], [], '.', color='tab:orange', ms=3, alpha=0.8)

    if desired_x is not None and desired_y is not None:
        ax.plot(desired_x, desired_y, color='lightgray', lw=1.0, alpha=0.7, label='Desired path')
        ax.legend(loc='best')

    def init():
        link_line.set_data([], [])
        path_line.set_data([], [])
        return link_line, path_line

    def update(i):
        xs = [0.0, x1[i], x2[i]]
        ys = [0.0, y1[i], y2[i]]
        link_line.set_data(xs, ys)
        path_line.set_data(x2[: i + 1], y2[: i + 1])
        return link_line, path_line

    ani = animation.FuncAnimation(fig, update, frames=n, init_func=init,
                                  interval=interval_ms, blit=False, repeat=False)

    if save_path:
        try:
            from matplotlib.animation import PillowWriter
            ani.save(save_path, writer=PillowWriter(fps=max(1, int(1000/interval_ms))))
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")

    plt.show()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the maze solver first.")
        # Create a dummy file for testing purposes if it doesn't exist?
        # Better to warn user.
        return

    # 1. Load Coords
    try:
        data = np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1, dtype=float)
        if data.size == 0:
            print("Error: Coords file is empty.")
            return
        if data.ndim == 1 and data.size >= 2:
            data = data.reshape(-1, 2)
        x_raw, y_raw = data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error loading {INPUT_FILE}: {e}")
        return

    # 2. Auto-Scaling Logic
    # We want the path to fit inside a box that is TARGET_WORKSPACE_RATIO * (A1+A2)
    
    # First, center the data at origin (bounding box center)
    min_x, max_x = x_raw.min(), x_raw.max()
    min_y, max_y = y_raw.min(), y_raw.max()
    
    width = max_x - min_x
    height = max_y - min_y
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    x_centered = x_raw - center_x
    y_centered = y_raw - center_y
    
    # Calculate scale factor
    # Max dimension of the path vs Target dimension
    max_path_dim = max(width, height)
    if max_path_dim == 0: max_path_dim = 1e-6
    
    target_dim_meters = (A1 + A2) * TARGET_WORKSPACE_RATIO
    scale = target_dim_meters / max_path_dim
    
    print(f"Auto-Scaling: input size {width:.1f}x{height:.1f} -> {target_dim_meters:.4f}m (Scale: {scale:.6f})")
    
    x_final = x_centered * scale
    y_final = y_centered * scale
    
    # 3. Compute IK
    q1, q2, r, feasible = compute_ik(x_final, y_final, A1, A2)
    
    n_feas = int(np.count_nonzero(feasible))
    print(f"Dimensions: {len(x_final)} points | Feasible: {n_feas}")
    
    # 4. Animate
    if ANIMATE:
        save_path = "output.gif" if SAVE_ANIM else None
        animate_arm(q1, q2, A1, A2, save_path=save_path, full_workspace=True, desired_x=x_final, desired_y=y_final)

if __name__ == '__main__':
    main()
