import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os

# FS data
#FSx = np.array([0.132596,0.125829,0.118014,0.112802,0.107589,0.102897,0.098723,0.096114,0.092981,0.090890,0.089320,0.087227,0.085656,0.085125,0.086157,0.086668,0.088742,0.091338,0.092373,0.096532,0.100692,0.105893,0.109013,0.115778,0.122026,0.126190,0.132438,0.138168,0.144418,0.148589,0.153280,0.156929,0.159539,0.162667,0.165279,0.165807,0.167894,0.168942,0.168949,0.169480,0.168969,0.165857,0.163781,0.160662,0.155463,0.149742,0.143497,0.135165,0.128914,0.123180,0.117966,0.113791,0.111178,0.109085,0.109595,0.110628,0.114788,0.117907,0.125195,0.129885,0.131972,0.134060,0.136668,0.137715,0.137722,0.135126,0.132005,0.129401])
#FSy = np.array([0.162056,0.163744,0.162636,0.160405,0.157895,0.155664,0.152313,0.149519,0.145328,0.141135,0.136941,0.132188,0.127435,0.121842,0.116805,0.111769,0.106731,0.102252,0.098614,0.094971,0.090770,0.087126,0.084324,0.081518,0.080110,0.078706,0.077578,0.078130,0.078121,0.080353,0.082304,0.084257,0.087051,0.089005,0.093197,0.096553,0.098788,0.102143,0.106060,0.111374,0.116689,0.123127,0.127046,0.130687,0.135450,0.139933,0.142460,0.143031,0.142480,0.139970,0.136901,0.132711,0.127959,0.122647,0.117052,0.112295,0.108933,0.105851,0.103603,0.104995,0.107230,0.109185,0.111699,0.114495,0.118132,0.122331,0.124573,0.124577])
FSy = np.array([0]*68)
FSx = np.array([-0.1908]*68)

def compute_ik(FSx: np.ndarray, FSy: np.ndarray, a1: float, a2: float):
    r = np.sqrt(FSx**2 + FSy**2)

    # Avoid division by zero where r==0
    r_safe = np.where(r == 0, 1e-12, r)

    # CosA and CosC terms
    CosA = (r**2 + a1**2 - a2**2) / (2.0 * r_safe * a1)
    CosC = (a1**2 + a2**2 - r**2) / (2.0 * a1 * a2)

    # Clamp to valid domain [-1, 1]
    CosA_clamped = np.clip(CosA, -1.0, 1.0)
    CosC_clamped = np.clip(CosC, -1.0, 1.0)

    SenA = np.sqrt(1.0 - CosA_clamped**2)
    SenC = np.sqrt(1.0 - CosC_clamped**2)

    # Angles using atan2(sin, cos)
    angleA = np.arctan2(SenA, CosA_clamped)
    angleC = np.arctan2(SenC, CosC_clamped)

    q1 = np.arctan2(FSy, FSx) - angleA
    q2 = np.pi - angleC

    # Feasibility masks (without clamping) for reporting
    feasible = (np.abs(CosA) <= 1.0) & (np.abs(CosC) <= 1.0) & (r > 0)

    return q1, q2, r, feasible


def animate_arm(
    q1: np.ndarray,
    q2: np.ndarray,
    a1: float,
    a2: float,
    save_path: str | None = None,
    full_workspace: bool = False,
    desired_x: np.ndarray | None = None,
    desired_y: np.ndarray | None = None,
    interval_ms: int = 100,
    hold_start_frames: int = 20,
):
    # Prepend a hold at (0,0) for the requested number of frames
    if hold_start_frames > 0:
        q1_ext = np.concatenate([np.zeros(hold_start_frames), q1])
        q2_ext = np.concatenate([np.zeros(hold_start_frames), q2])
    else:
        q1_ext, q2_ext = q1, q2

    n = len(q1_ext)
    x1 = a1 * np.cos(q1_ext)
    y1 = a1 * np.sin(q1_ext)
    x2 = x1 + a2 * np.cos(q1_ext + q2_ext)
    y2 = y1 + a2 * np.sin(q1_ext + q2_ext)

    L = a1 + a2
    # Determine axis limits from both reachable workspace and actual path
    margin = 0.02
    if full_workspace:
        x_min, x_max = -L, L
        y_min, y_max = -L, L
    else:
        x_min = min(-L, 0.0, float(x1.min()), float(x2.min()))
        x_max = max(L, 0.0, float(x1.max()), float(x2.max()))
        y_min = min(-L, 0.0, float(y1.min()), float(y2.min()))
        y_max = max(L, 0.0, float(y1.max()), float(y2.max()))
        # Also include provided FS path if available (globals FSx, FSy)
        try:
            x_min = min(x_min, float(FSx.min()))
            x_max = max(x_max, float(FSx.max()))
            y_min = min(y_min, float(FSy.min()))
            y_max = max(y_max, float(FSy.max()))
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title('2-DOF Arm Simulation')

    link_line, = ax.plot([], [], '-o', lw=3, ms=6)
    path_line, = ax.plot([], [], '.', color='tab:orange', ms=3, alpha=0.8)
    # Draw desired path for reference (either provided or fall back to FS path)
    try:
        if desired_x is not None and desired_y is not None:
            ax.plot(desired_x, desired_y, color='lightgray', lw=1.0, alpha=0.7, label='Desired path')
        else:
            ax.plot(FSx, FSy, color='lightgray', lw=1.0, alpha=0.7, label='Desired path')
        ax.legend(loc='best')
    except Exception:
        pass

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

    # Disable blitting to avoid partial draws on some Windows backends
    ani = animation.FuncAnimation(fig, update, frames=n, init_func=init, interval=interval_ms, blit=False, repeat=False)

    if save_path:
        try:
            from matplotlib.animation import PillowWriter
            ani.save(save_path, writer=PillowWriter(fps=10))
        except Exception as e:
            print(f"Could not save animation: {e}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate 2DOF planar IK for FSx/FSy points")
    parser.add_argument("--a1", type=float, default=0.0848, help="Link 1 length (meters)")
    parser.add_argument("--a2", type=float, default=0.106, help="Link 2 length (meters)")
    parser.add_argument("--save", action="store_true", help="Save plot to ik_angles.png")
    parser.add_argument("--animate", action="store_true", help="Animate the 2-DOF arm following q1,q2")
    parser.add_argument("--save_anim", type=str, default="", help="Path to save GIF animation (e.g., arm.gif)")
    parser.add_argument("--full_workspace", action="store_true", help="Force axes to full reachable workspace [-L,L]")
    parser.add_argument("--demo_linear", action="store_true", help="Demo: linearly interpolate from origin to center of FS path")
    parser.add_argument("--steps", type=int, default=80, help="Number of steps for demo paths")
    parser.add_argument("--demo_circle", action="store_true", help="Demo: follow the specified circle path")
    parser.add_argument("--circle_steps", type=int, default=180, help="Number of samples around the circle")
    parser.add_argument("--demo_flower", action="store_true", help="Demo: follow the specified flower path")
    parser.add_argument("--flower_steps", type=int, default=180, help="Number of samples for the flower path")
    parser.add_argument("--interval_ms", type=int, default=100, help="Animation interval per frame in milliseconds")
    parser.add_argument("--hold_start_s", type=float, default=2.0, help="Seconds to hold at q1=q2=0 before starting")
    # Maze integration
    parser.add_argument("--path_csv", type=str, default="", help="CSV file with x,y columns (pixels) for maze path")
    parser.add_argument("--path_json", type=str, default="", help="JSON file: list of [x,y] points (pixels)")
    parser.add_argument("--pixel_scale", type=float, default=0.001, help="Meters per pixel to scale maze path")
    parser.add_argument("--start_at_origin", action="store_true", help="Translate path so first point maps to (0,0)")
    args = parser.parse_args()

    a1, a2 = args.a1, args.a2
    interval_ms = max(1, int(args.interval_ms))
    hold_frames = max(0, int(round(args.hold_start_s * 1000.0 / interval_ms)))

    # Helper: load path from CSV or JSON (pixels)
    def load_path_pixels():
        if args.path_csv:
            path_file = args.path_csv
            try:
                data = np.loadtxt(path_file, delimiter=",", dtype=float)
                if data.ndim == 1 and data.size >= 2:
                    data = data.reshape(-1, 2)
                if data.shape[1] < 2:
                    raise ValueError("CSV must have at least 2 columns: x,y")
                return data[:,0], data[:,1]
            except Exception as e:
                print(f"Failed to read CSV path '{path_file}': {e}")
                return None, None
        if args.path_json:
            path_file = args.path_json
            try:
                with open(path_file, 'r') as f:
                    pts = json.load(f)
                arr = np.array(pts, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    raise ValueError("JSON must be a list of [x,y] pairs")
                return arr[:,0], arr[:,1]
            except Exception as e:
                print(f"Failed to read JSON path '{path_file}': {e}")
                return None, None
        return None, None

    # If a maze path is provided, consume it first
    if args.path_csv or args.path_json:
        px, py = load_path_pixels()
        if px is None:
            return
        # Normalize: set start to origin if requested
        if args.start_at_origin:
            px = px - px[0]
            py = py - py[0]
        # Scale to meters
        s = float(args.pixel_scale)
        demo_x = px * s
        demo_y = py * s

        q1, q2, r, feasible = compute_ik(demo_x, demo_y, a1, a2)
        n_feasible = int(np.count_nonzero(feasible))
        print(f"Maze path loaded: {len(demo_x)} points | Feasible without clamp: {n_feasible}")
        if args.animate or args.save_anim:
            save_path = args.save_anim if args.save_anim else None
            animate_arm(q1, q2, a1, a2, save_path, full_workspace=args.full_workspace, desired_x=demo_x, desired_y=demo_y, interval_ms=interval_ms, hold_start_frames=hold_frames)
        else:
            print("Tip: add --animate to see the arm follow the maze path.")
        return

    # If demo requested: build a straight-line path from origin to the center of FS
    if args.demo_linear:
        cx, cy = float(FSx.mean()), float(FSy.mean())
        t = np.linspace(0.0, 1.0, num=max(2, args.steps))
        demo_x = t * cx
        demo_y = t * cy
        q1, q2, r, feasible = compute_ik(demo_x, demo_y, a1, a2)
        # Report and animate, then exit
        q1_deg = np.degrees(q1)
        q2_deg = np.degrees(q2)
        n_feasible = int(np.count_nonzero(feasible))
        print(f"Demo linear to center ({cx:.4f},{cy:.4f}) | Steps: {len(demo_x)} | Feasible without clamp: {n_feasible}")
        if args.animate or args.save_anim:
            save_path = args.save_anim if args.save_anim else None
            animate_arm(q1, q2, a1, a2, save_path, full_workspace=args.full_workspace, desired_x=demo_x, desired_y=demo_y)
        else:
            print("Tip: add --animate to see the arm move along the line to center.")
        return

    # If circle demo requested: use provided parametric equations
    if args.demo_circle:
        steps = max(8, args.circle_steps)
        # Match: for (i=0; i<t; ++i) rad = radians(i); with t=steps
        rad = 2.0 * np.pi * (np.arange(steps) / float(steps))
        demo_x = -0.07 + 0.04 * np.cos(rad)
        demo_y =  0.13 + 0.04 * np.sin(rad)
        q1, q2, r, feasible = compute_ik(demo_x, demo_y, a1, a2)
        n_feasible = int(np.count_nonzero(feasible))
        print(f"Demo circle center(-0.07,0.13), R=0.04 | Steps: {steps} | Feasible without clamp: {n_feasible}")
        if args.animate or args.save_anim:
            save_path = args.save_anim if args.save_anim else None
            animate_arm(q1, q2, a1, a2, save_path, full_workspace=args.full_workspace, desired_x=demo_x, desired_y=demo_y, interval_ms=interval_ms, hold_start_frames=hold_frames)
        else:
            print("Tip: add --animate to see the arm move along the circle.")
        return

    # If flower demo requested: use provided parametric equations
    if args.demo_flower:
        steps = max(8, args.flower_steps)
        # for (i=0; i<t; ++i) rad = radians(i); with t=steps
        rad = 2.0 * np.pi * (np.arange(steps) / float(steps))
        demo_x = -0.06 + 0.04 * (1 + np.sin(5 * rad) * np.cos(rad))
        demo_y =  0.10 + 0.04 * (1 + np.sin(5 * rad) * np.sin(rad))
        q1, q2, r, feasible = compute_ik(demo_x, demo_y, a1, a2)
        n_feasible = int(np.count_nonzero(feasible))
        print(f"Demo flower center(-0.06,0.10) | Steps: {steps} | Feasible without clamp: {n_feasible}")
        if args.animate or args.save_anim:
            save_path = args.save_anim if args.save_anim else None
            animate_arm(q1, q2, a1, a2, save_path, full_workspace=args.full_workspace, desired_x=demo_x, desired_y=demo_y, interval_ms=interval_ms, hold_start_frames=hold_frames)
        else:
            print("Tip: add --animate to see the arm move along the flower.")
        return

    # Apply FSx[i] -= 0.18 for i in [0, 67) if x_offset provided
    x_offset = getattr(args, 'x_offset', None)
    if x_offset is None:
        # add the argument dynamically if not present (backward compat)
        x_offset = -0.18
    FSx_use = FSx.copy()
    n_off = min(67, FSx_use.shape[0])
    FSx_use[:n_off] = FSx_use[:n_off] + x_offset

    q1, q2, r, feasible = compute_ik(FSx_use, FSy, a1, a2)

    q1_deg = np.degrees(q1)
    q2_deg = np.degrees(q2)

    n_feasible = int(np.count_nonzero(feasible))
    print(f"Points: {len(FSx)} | Feasible without clamp: {n_feasible} | With clamp: {len(FSx)}")
    print(f"a1={a1:.4f}, a2={a2:.4f}")
    print("First 5 q1 (deg):", np.round(q1_deg[:5], 2))
    print("First 5 q2 (deg):", np.round(q2_deg[:5], 2))

    idx = np.arange(len(FSx))
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(idx, q1_deg, label='q1 (deg)')
    axs[0].set_ylabel('q1 [deg]')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(idx, q2_deg, label='q2 (deg)', color='tab:orange')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('q2 [deg]')
    axs[1].grid(True, alpha=0.3)

    fig.suptitle('IK Angles from FS trajectory')
    fig.tight_layout()

    if args.save:
        fig.savefig('ik_angles.png', dpi=200)

    # If animating, don't block on the static plot
    if args.animate or args.save_anim:
        plt.close(fig)
        save_path = args.save_anim if args.save_anim else None
        animate_arm(q1, q2, a1, a2, save_path, full_workspace=args.full_workspace, desired_x=FSx_use, desired_y=FSy)
    else:
        plt.show()


if __name__ == "__main__":
    main()
