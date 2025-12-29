import cv2
import numpy as np
import sys
import os

# Add project root to path so we can import from maze package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maze import maze_pipeline as mp
from maze import maze_solver as ms

# --- CONSTANTS ---
IMAGE_PATH = "maze.jpg"   # User: make sure this file exists in the current dir or update path
OUTPUT_CSV = "coords.csv"
ROBOT_WAYPOINT_STEP = 15

def main():
    # 1. Load Image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found. Please provide a maze image.")
        return

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load {IMAGE_PATH}")
        return

    # 2. Select Corners
    print("--- Corner Selection ---")
    print("Click 4 corners. Press 'q' to quit.")
    
    mp.corners = []
    mp.img_display = img.copy()
    
    cv2.namedWindow("Select Corners")
    cv2.setMouseCallback("Select Corners", mp.mouse_callback)
    cv2.imshow("Select Corners", mp.img_display)
    
    while len(mp.corners) < 4:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
            
    print("Corners collected.")
    cv2.destroyWindow("Select Corners")
    
    # 3. Warp
    sorted_pts = mp.order_points(mp.corners)
    warped_view = mp.warp_perspective(img, sorted_pts)
    
    # 4. Skeletonize
    _, skeleton_map = mp.process_pipeline(warped_view)
    
    # 5. Select Start/End
    ms.start_point = None
    ms.end_point = None
    ms.selecting_start = True
    ms.img_display = cv2.cvtColor(skeleton_map, cv2.COLOR_GRAY2BGR)
    
    print("--- Start/End Selection ---")
    print("Click Green for Start, Red for End. Press SPACE to solve.")
    
    cv2.namedWindow("Select Start/End")
    cv2.setMouseCallback("Select Start/End", ms.mouse_click, skeleton_map)
    cv2.imshow("Select Start/End", ms.img_display)
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == 32: # Space
            if ms.start_point and ms.end_point:
                break
            else:
                print("Set both points first!")
        if key == 27: # ESC
            cv2.destroyAllWindows()
            return
            
    cv2.destroyWindow("Select Start/End")
    
    # 6. Solve
    print("Solving...")
    raw_path = ms.solve_bfs(skeleton_map, ms.start_point, ms.end_point)
    if not raw_path:
        print("No path found!")
        return
        
    # 7. Simplify
    robot_path = ms.simplify_path(raw_path, step=ROBOT_WAYPOINT_STEP)
    print(f"Path found. Raw: {len(raw_path)}, Robot Waypoints: {len(robot_path)}")
    
    # 8. Export Coords
    # Save as x,y
    with open(OUTPUT_CSV, 'w') as f:
        f.write("x,y\n")
        for x, y in robot_path:
            f.write(f"{x},{y}\n")
            
    print(f"Saved coordinates to {OUTPUT_CSV}")
    
    # 9. Visualization (Optional - wait for key)
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
