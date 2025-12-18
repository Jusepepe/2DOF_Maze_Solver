import cv2
import numpy as np

import maze_pipeline as mp
import maze_solver as ms


def main():
    # 1) Load the original photo
    img = cv2.imread('maze7.png')
    if img is None:
        print("Error: maze_photo4.jpeg not found in current directory.")
        return

    # 2) Corner selection using maze_pipeline's UI callback
    mp.corners = []
    mp.img_display = img.copy()

    print("--- Corner Selection ---")
    print("Click the 4 corners of the maze in any order, then press any key.")

    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", mp.mouse_callback)
    cv2.imshow("Select 4 Corners", mp.img_display)

    # Wait until 4 corners are collected
    while True:
        key = cv2.waitKey(10) & 0xFF
        if len(mp.corners) == 4:
            print("4 points collected. Press any key to continue...")
            cv2.waitKey(0)
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select 4 Corners")

    # 3) Warp perspective
    sorted_pts = mp.order_points(mp.corners)
    warped_view = mp.warp_perspective(img, sorted_pts)

    # 4) Process pipeline to get skeletonized path mask (single-channel, 0/255)
    _, skeleton_map = mp.process_pipeline(warped_view)

    # 5) Start/End selection using maze_solver's UI callback
    ms.start_point = None
    ms.end_point = None
    ms.selecting_start = True
    # Provide a BGR image for drawing during start/end selection using the skeleton as background
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
            return

    # 6) Solve via BFS on binary map (white=255 as path)
    print("Solving maze...")
    raw_path = ms.solve_bfs(skeleton_map, ms.start_point, ms.end_point)

    if raw_path is None:
        print("No path found. Ensure start/end are on the white path.")
        cv2.destroyAllWindows()
        return

    print(f"Path found! Length: {len(raw_path)} pixels")

    # 7) Simplify for robot waypoints
    robot_path = ms.simplify_path(raw_path, step=ms.ROBOT_WAYPOINT_STEP)
    print(f"Robot Waypoints: {len(robot_path)}")

    # 8) Visualize solution directly on the warped (color) image
    vis = warped_view.copy()
    for p in raw_path:
        vis[p[1], p[0]] = (255, 0, 0)  # Blue path

    for i in range(len(robot_path) - 1):
        pt1 = robot_path[i]
        pt2 = robot_path[i + 1]
        cv2.line(vis, pt1, pt2, (0, 255, 255), 2)  # Yellow waypoint lines
        cv2.circle(vis, pt1, 3, (0, 165, 255), -1)  # Orange waypoint dots

    cv2.imshow("Warped View", warped_view)
    cv2.imshow("Skeleton Map", skeleton_map)
    cv2.imshow("Solution", vis)

    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
