import argparse
import json
import numpy as np
import cv2
import os
import sys

# Make project root importable when running this script directly
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from maze import maze_pipeline as mp
from maze import maze_solver as ms
from robot.ik_from_coords import compute_ik, animate_arm


def select_corners(image_bgr):
    mp.corners = []
    mp.img_display = image_bgr.copy()

    print("--- Corner Selection ---")
    print("Click the 4 corners of the maze in any order, then press any key.")

    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", mp.mouse_callback)
    cv2.imshow("Select 4 Corners", mp.img_display)

    while True:
        key = cv2.waitKey(10) & 0xFF
        if len(mp.corners) == 4:
            print("4 points collected. Press any key to continue...")
            cv2.waitKey(0)
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit(0)

    cv2.destroyWindow("Select 4 Corners")
    return mp.order_points(mp.corners)


def select_start_end(skeleton_map):
    ms.start_point = None
    ms.end_point = None
    ms.selecting_start = True
    ms.img_display = cv2.cvtColor(skeleton_map, cv2.COLOR_GRAY2BGR)

    print("--- Start/End Selection ---")
    print("1. Click the START point (Green).")
    print("2. Click the END point (Red).")
    print("3. Press SPACE to solve. Press ESC to cancel.")

    cv2.imshow("Select Start & End", ms.img_display)
    cv2.setMouseCallback("Select Start & End", ms.mouse_click, skeleton_map)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space
            if ms.start_point and ms.end_point:
                break
            else:
                print("Please select both Start and End points first.")
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            raise SystemExit(0)

    return ms.start_point, ms.end_point


def main():
    p = argparse.ArgumentParser(description="Select image, solve maze, and simulate 2DOF arm following the path.")
    p.add_argument('--image', type=str, required=True, help='Path to maze image')
    p.add_argument('--pixel_threshold', type=int, default=200, help='Threshold for binarization (0-255)')

    # Arm and mapping options
    p.add_argument('--a1', type=float, default=0.0848, help='Link 1 length (m)')
    p.add_argument('--a2', type=float, default=0.1060, help='Link 2 length (m)')
    p.add_argument('--scale', type=float, default=0.001, help='Meters per pixel')
    p.add_argument('--auto_scale', action='store_true', help='Automatically scale path to fit inside arm reach')
    p.add_argument('--reach_ratio', type=float, default=0.95, help='Fraction of max reach (a1+a2) used when auto-scaling')
    p.add_argument('--start_at_origin', action='store_true', help='Translate path so the first point maps to (0,0)')
    p.add_argument('--rotate_deg', type=float, default=0.0, help='Rotate path around origin (degrees)')

    # Animation
    p.add_argument('--animate', action='store_true', help='Animate the arm following the solved path')
    p.add_argument('--full_workspace', action='store_true', help='Show full reachable workspace in plot')
    p.add_argument('--interval_ms', type=int, default=100, help='Frame interval (ms)')
    p.add_argument('--hold_start_s', type=float, default=2.0, help='Hold at q1=q2=0 before moving (s)')
    p.add_argument('--save_anim', type=str, default='', help='Output GIF path')

    # Outputs
    p.add_argument('--save_path_csv', type=str, default='', help='Save simplified path (meters) to CSV')
    p.add_argument('--save_angles_csv', type=str, default='', help='Save q1,q2 (deg) to CSV')

    args = p.parse_args()

    # 1) Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: image not found: {args.image}")
        return

    # 2) Corner selection and warp
    sorted_pts = select_corners(img)
    warped_view = mp.warp_perspective(img, sorted_pts)

    # 3) Process to skeleton
    # Use pipeline but override threshold if provided by adjusting source
    _, skeleton_map = mp.process_pipeline(warped_view)

    # 4) Start/End selection
    start, end = select_start_end(skeleton_map)

    # 5) Solve BFS on binary map and simplify
    print("Solving maze...")
    raw_path = ms.solve_bfs(skeleton_map, start, end)
    if raw_path is None:
        print("No path found. Ensure start/end are on the white path.")
        cv2.destroyAllWindows()
        return

    robot_path = ms.simplify_path(raw_path, step=ms.ROBOT_WAYPOINT_STEP)
    print(f"Path found. Raw: {len(raw_path)} pts | Waypoints: {len(robot_path)}")

    # 6) Convert to meters and frame-transform
    px = np.array([p[0] for p in robot_path], dtype=float)
    py = np.array([p[1] for p in robot_path], dtype=float)

    if args.start_at_origin:
        px = px - px[0]
        py = py - py[0]

    if args.rotate_deg:
        th = np.deg2rad(args.rotate_deg)
        xr = px*np.cos(th) - py*np.sin(th)
        yr = px*np.sin(th) + py*np.cos(th)
        px, py = xr, yr

    # Hard-code: mirror Y only (leave X as-is)
    py = -py

    # Optional: auto-scale path to fit inside workspace radius reach_ratio*(a1+a2)
    if args.auto_scale:
        r_pix = np.sqrt(px**2 + py**2)
        r_max_pix = float(np.max(r_pix)) if r_pix.size > 0 else 0.0
        if r_max_pix > 0:
            L_eff = float(args.reach_ratio) * (args.a1 + args.a2)
            auto_scale = L_eff / r_max_pix
            if auto_scale <= 0:
                print("Warning: computed non-positive auto scale; falling back to --scale")
            else:
                print(f"Auto scale computed: {auto_scale:.6f} m/px (reach_ratio={args.reach_ratio}, L_eff={L_eff:.4f} m)")
                args.scale = auto_scale
        else:
            print("Warning: path radius is zero; auto_scale skipped")

    x = px * args.scale
    y = py * args.scale

    # Translate to the second quadrant (x < 0, y > 0)
    if x.size > 0 and y.size > 0:
        margin = 0.0
        # Shift X so its maximum is at -margin (entire path x <= -margin)
        dx = -(np.max(x) + margin)
        # Shift Y so its minimum is at +margin (entire path y >= margin)
        dy = -(np.min(y)) + margin
        x = x + dx
        y = y + dy

        # Mirror in X within the second quadrant (reflect about path's vertical center)
        x_center = 0.5 * (np.min(x) + np.max(x))
        x = 2.0 * x_center - x

    if args.save_path_csv:
        np.savetxt(args.save_path_csv, np.column_stack([x, y]), delimiter=',', header='x_m,y_m', comments='')
        print(f"Saved path (m) to {args.save_path_csv}")

    # 7) IK
    q1, q2, r, feasible = compute_ik(x, y, args.a1, args.a2)
    q1_deg = np.degrees(q1)
    q2_deg = np.degrees(q2)
    print(f"Feasible (no clamp): {int(np.count_nonzero(feasible))} / {len(x)}")

    if args.save_angles_csv:
        np.savetxt(args.save_angles_csv, np.column_stack([q1_deg, q2_deg]), delimiter=',', header='q1_deg,q2_deg', comments='')
        print(f"Saved angles (deg) to {args.save_angles_csv}")

    # 8) Animate
    if args.animate or args.save_anim:
        save_path = args.save_anim if args.save_anim else None
        animate_arm(q1, q2, args.a1, args.a2,
                    save_path=save_path,
                    full_workspace=args.full_workspace,
                    desired_x=x, desired_y=y,
                    interval_ms=args.interval_ms,
                    hold_start_s=args.hold_start_s)

    # 9) Show visualization windows from maze stages until a keypress
    cv2.imshow("Warped View", warped_view)
    vis = warped_view.copy()
    for p in raw_path:
        vis[p[1], p[0]] = (255, 0, 0)
    for i in range(len(robot_path) - 1):
        pt1 = robot_path[i]
        pt2 = robot_path[i + 1]
        cv2.line(vis, pt1, pt2, (0, 255, 255), 2)
        cv2.circle(vis, pt1, 3, (0, 165, 255), -1)
    cv2.imshow("Solution", vis)
    print("Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
