import cv2
import numpy as np
import os
import sys

# Ensure we can import from the 'maze' directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from maze import maze_pipeline as mp
from maze import maze_solver as ms
import simulations
from ik_from_coords import compute_ik, animate_arm, A1, A2

# --- CONSTANTS ---
CAMERA_INDEX = 0    # Default webcam
ROBOT_WAYPOINT_STEP = 15
SAVE_COORDS_CSV = "coords_camera.csv"
ANIMATE = True

# Safe Box in Right Quadrant (X>0, Y>0)
# Robot reach is approx 0.19m. 
# We target a box [0.03, 0.12] for both X and Y.
# This ensures r is roughly 0.04 to 0.17, well within reach.
TARGET_BOX_MIN = 0.03
TARGET_BOX_MAX = 0.12

def main():
    # 1. Capture from Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        return

    print("--- CAMERA FEED ---")
    print("Press 'c' to CAPTURE the maze image.")
    print("Press 'q' to QUIT.")

    img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        cv2.imshow("Camera - Press 'c' to capture", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):
            img = frame.copy()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyWindow("Camera - Press 'c' to capture")
    
    if img is None:
        return

    # 2. Corner Selection
    print("--- Corner Selection ---")
    mp.corners = []
    mp.img_display = img.copy()
    
    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", mp.mouse_callback)
    cv2.imshow("Select 4 Corners", mp.img_display)
    
    while len(mp.corners) < 4:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select 4 Corners")
    
    # 3. Warp & Vision Pipeline
    sorted_pts = mp.order_points(mp.corners)
    warped_view = mp.warp_perspective(img, sorted_pts)
    _, skeleton_map = mp.process_pipeline(warped_view)
    
    # 4. Start/End Selection
    ms.start_point = None
    ms.end_point = None
    ms.selecting_start = True
    ms.img_display = cv2.cvtColor(skeleton_map, cv2.COLOR_GRAY2BGR)
    
    cv2.namedWindow("Select Start/End")
    cv2.setMouseCallback("Select Start/End", ms.mouse_click, skeleton_map)
    cv2.imshow("Select Start/End", ms.img_display)
    
    print("Select Green (Start) then Red (End). Press SPACE to solve.")
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == 32: # Space
            if ms.start_point and ms.end_point:
                break
            else:
                print("Select both points first.")
        elif key == 27: # ESC
            cv2.destroyAllWindows()
            return
            
    cv2.destroyWindow("Select Start/End")
    
    # 5. Solve Maze
    raw_path = ms.solve_bfs(skeleton_map, ms.start_point, ms.end_point)
    if not raw_path:
        print("No path found.")
        return

    robot_path = ms.simplify_path(raw_path, step=ROBOT_WAYPOINT_STEP)
    print(f"Path found: {len(robot_path)} waypoints")

    # 6. Coordinate Mapping -> Right Quadrant Fitting
    # We want to map the pixel path into the box [TARGET_BOX_MIN, TARGET_BOX_MAX]
    
    # Extract raw pixels
    x_img = np.array([p[0] for p in robot_path], dtype=float)
    y_img = np.array([p[1] for p in robot_path], dtype=float)
    
    # 1. Normalize to [0, 1] based on bounds
    min_pix_x, max_pix_x = x_img.min(), x_img.max()
    min_pix_y, max_pix_y = y_img.min(), y_img.max()
    
    w_pix = max_pix_x - min_pix_x
    h_pix = max_pix_y - min_pix_y
    if w_pix == 0: w_pix = 1
    if h_pix == 0: h_pix = 1
    
    # Preserve aspect ratio by using max dimension
    scale_norm = 1.0 / max(w_pix, h_pix)
    
    norm_x = (x_img - min_pix_x) * scale_norm
    norm_y = (y_img - min_pix_y) * scale_norm
    
    # 2. Scale to Physical Dimensions
    # Desired Width/Height in meters
    target_size = TARGET_BOX_MAX - TARGET_BOX_MIN
    
    x_scaled = norm_x * target_size
    y_scaled = norm_y * target_size
    
    # 3. Position in Right Quadrant
    # Invert Y to match Cartesian (Image Y is down, we want "up" behavior?)
    # Usually maze start is top-left. If we want that to be "far" or "close"?
    # Let's map Image Top (Low Y) to Physical Top (High Y) of the box.
    y_scaled = target_size - y_scaled # Invert [0, size] -> [size, 0]
    
    x_robot = x_scaled + TARGET_BOX_MIN
    y_robot = y_scaled + TARGET_BOX_MIN
    
    # Print stats
    print(f"Mapped to box: X[{x_robot.min():.3f}, {x_robot.max():.3f}], Y[{y_robot.min():.3f}, {y_robot.max():.3f}]")

    # Save CSV
    if SAVE_COORDS_CSV:
        np.savetxt(SAVE_COORDS_CSV, np.column_stack([x_robot, y_robot]), 
                   delimiter=',', header='x,y', comments='')
        print(f"Saved coords to {SAVE_COORDS_CSV}")

    # 7. Compute IK & Animate
    q1, q2, r, feasible = compute_ik(x_robot, y_robot, A1, A2)
    n_feas = int(np.count_nonzero(feasible))
    print(f"Feasible Points: {n_feas} / {len(x_robot)}")
    
    if ANIMATE:
        animate_arm(q1, q2, A1, A2, full_workspace=True, desired_x=x_robot, desired_y=y_robot)

    # Show result on image (wait for key)
    vis = warped_view.copy()
    for i in range(len(robot_path) - 1):
        pt1 = robot_path[i]
        pt2 = robot_path[i+1]
        cv2.line(vis, pt1, pt2, (0, 255, 255), 2)
        cv2.circle(vis, pt1, 3, (0, 165, 255), -1)
    
    cv2.imshow("Solution", vis)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
