import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
                 save_path: str | None = None, full_workspace: bool = False,
                 desired_x: np.ndarray | None = None,
                 desired_y: np.ndarray | None = None,
                 interval_ms: int = 100, hold_start_s: float = 2.0):
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
        x_min = min(-L, 0.0, float(x1.min()), float(x2.min()))
        x_max = max(L, 0.0, float(x1.max()), float(x2.max()))
        y_min = min(-L, 0.0, float(y1.min()), float(y2.min()))
        y_max = max(L, 0.0, float(y1.max()), float(y2.max()))
        if desired_x is not None and desired_y is not None:
            x_min = min(x_min, float(np.min(desired_x)))
            x_max = max(x_max, float(np.max(desired_x)))
            y_min = min(y_min, float(np.min(desired_y)))
            y_max = max(y_max, float(np.max(desired_y)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title('2-DOF Arm Simulation (from coordinates)')

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
        except Exception as e:
            print(f"Could not save animation: {e}")

    plt.show()


def load_coords(args):
    if args.path_csv:
        data = np.loadtxt(args.path_csv, delimiter=",", dtype=float)
        if data.ndim == 1 and data.size >= 2:
            data = data.reshape(-1, 2)
        if data.shape[1] < 2:
            raise ValueError("CSV must have at least 2 columns: x,y")
        return data[:, 0], data[:, 1]
    if args.path_json:
        with open(args.path_json, 'r') as f:
            pts = json.load(f)
        arr = np.array(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("JSON must be a list of [x,y] pairs")
        return arr[:, 0], arr[:, 1]
    if args.xy:
        # parse "x1,y1;x2,y2;..."
        pairs = []
        for seg in args.xy.split(';'):
            if not seg.strip():
                continue
            x_str, y_str = seg.split(',')
            pairs.append((float(x_str), float(y_str)))
        arr = np.array(pairs, dtype=float)
        return arr[:, 0], arr[:, 1]
    raise ValueError("No coordinates provided. Use --path_csv, --path_json, or --xy")


def main():
    p = argparse.ArgumentParser(description="Compute IK for a list of 2D coordinates (meters) and optionally animate.")
    p.add_argument('--a1', type=float, default=0.0848, help='Link 1 length (m)')
    p.add_argument('--a2', type=float, default=0.1060, help='Link 2 length (m)')

    # Inputs
    p.add_argument('--path_csv', type=str, default='', help='CSV file with x,y columns (meters by default)')
    p.add_argument('--path_json', type=str, default='', help='JSON file with [[x,y],...] (meters by default)')
    p.add_argument('--xy', type=str, default='', help='Inline coordinates: "x1,y1;x2,y2;..." (meters)')

    # Preprocessing
    p.add_argument('--start_at_origin', action='store_true', help='Translate so first point is (0,0)')
    p.add_argument('--invert_y', action='store_true', help='Invert Y (pixels-down to Cartesian-up)')
    p.add_argument('--rotate_deg', type=float, default=0.0, help='Rotate coordinates by degrees about origin')
    p.add_argument('--scale', type=float, default=1.0, help='Scale factor applied to all coordinates')

    # Animation/output
    p.add_argument('--animate', action='store_true', help='Animate the arm following the path')
    p.add_argument('--interval_ms', type=int, default=100, help='Animation frame interval (ms)')
    p.add_argument('--hold_start_s', type=float, default=2.0, help='Hold at q1=q2=0 (s) before moving')
    p.add_argument('--full_workspace', action='store_true', help='Axis limits set to full reach')
    p.add_argument('--save_anim', type=str, default='', help='Path to save GIF animation')
    p.add_argument('--out_angles_csv', type=str, default='', help='Write q1,q2 in degrees to CSV')

    args = p.parse_args()

    # Load coordinates
    x, y = load_coords(args)

    # Preprocess
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if args.start_at_origin:
        x = x - x[0]
        y = y - y[0]

    if args.invert_y:
        y = -y

    if args.rotate_deg:
        th = np.deg2rad(args.rotate_deg)
        xr = x*np.cos(th) - y*np.sin(th)
        yr = x*np.sin(th) + y*np.cos(th)
        x, y = xr, yr

    if args.scale != 1.0:
        x *= args.scale
        y *= args.scale

    # IK
    q1, q2, r, feasible = compute_ik(x, y, args.a1, args.a2)
    q1_deg = np.degrees(q1)
    q2_deg = np.degrees(q2)

    n = len(x)
    n_feas = int(np.count_nonzero(feasible))
    print(f"Points: {n} | Feasible without clamp: {n_feas}")
    print("First 5 q1 (deg):", np.round(q1_deg[:5], 2))
    print("First 5 q2 (deg):", np.round(q2_deg[:5], 2))

    if args.out_angles_csv:
        out = np.column_stack([q1_deg, q2_deg])
        np.savetxt(args.out_angles_csv, out, delimiter=',', header='q1_deg,q2_deg', comments='')
        print(f"Saved angles to {args.out_angles_csv}")

    if args.animate or args.save_anim:
        save_path = args.save_anim if args.save_anim else None
        animate_arm(q1, q2, args.a1, args.a2, save_path,
                    full_workspace=args.full_workspace,
                    desired_x=x, desired_y=y,
                    interval_ms=args.interval_ms,
                    hold_start_s=args.hold_start_s)


if __name__ == '__main__':
    main()
