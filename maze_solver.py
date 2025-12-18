import cv2
import numpy as np
from collections import deque

# --- CONFIGURATION ---
# How many pixels to skip between waypoints for the robot
# Higher = smoother, faster robot motion but less precision
# Lower = extremely accurate but slow, jittery motion
ROBOT_WAYPOINT_STEP = 15 

# Global variables for mouse interaction
start_point = None
end_point = None
selecting_start = True # Flag to track if we are picking start or end

def find_nearest_white_pixel(img, target):
    """
    If the user clicks slightly on a wall, this finds the nearest white path pixel.
    """
    nonzero = cv2.findNonZero(img)
    if nonzero is None:
        return None
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return (nonzero[nearest_index][0][0], nonzero[nearest_index][0][1])

def mouse_click(event, x, y, flags, param):
    global start_point, end_point, selecting_start, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Snap click to nearest valid path pixel
        # We assume the 'path_mask' is passed via param
        path_mask = param
        clicked_point = (x, y)
        valid_point = find_nearest_white_pixel(path_mask, clicked_point)
        
        if valid_point:
            if selecting_start:
                start_point = valid_point
                print(f"Start Point set: {start_point}")
                cv2.circle(img_display, start_point, 5, (0, 255, 0), -1) # Green for Start
                selecting_start = False
            else:
                end_point = valid_point
                print(f"End Point set: {end_point}")
                cv2.circle(img_display, end_point, 5, (0, 0, 255), -1) # Red for End
            
            cv2.imshow("Select Start & End", img_display)

def solve_bfs(binary_img, start, end):
    """
    Standard Breadth-First Search to find shortest path.
    Returns a list of (x, y) tuples.
    """
    h, w = binary_img.shape
    queue = deque([[start]]) # Queue stores paths
    visited = set()
    visited.add(start)
    
    # Directions: Up, Down, Left, Right
    # Add diagonals [(1,1), (-1,-1)...] if you want diagonal movement
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),(1,1), (-1,-1)]
    
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        
        if (x, y) == end:
            return path
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < w and 0 <= ny < h:
                # Check if pixel is white (path) and not visited
                if binary_img[ny, nx] == 255 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    queue.append(new_path)
                    
    return None # No path found

def simplify_path(path, step=10):
    """
    Reduces the dense list of pixels into sparse waypoints for the robot.
    """
    if not path: return []
    
    waypoints = [path[0]] # Always keep start
    
    # Take every Nth point
    for i in range(1, len(path) - 1):
        if i % step == 0:
            waypoints.append(path[i])
            
    waypoints.append(path[-1]) # Always keep end
    return waypoints

# --- MAIN ---
if __name__ == "__main__":
    # 1. Load the skeletonized/binary image from the previous script
    # It MUST be 1 channel (grayscale)
    filename = "skeleton_output.png" # Make sure to save your previous result!
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Run the vision script first and save the result as 'skeleton_output.png'")
        exit()

    # Create a color version just for display
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 2. Setup Mouse Interface
    print("--- INSTRUCTIONS ---")
    print("1. Click the START point (Green).")
    print("2. Click the END point (Red).")
    print("3. Press SPACE to solve.")
    
    cv2.imshow("Select Start & End", img_display)
    cv2.setMouseCallback("Select Start & End", mouse_click, img) # Pass binary img as param

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 32: # Spacebar
            if start_point and end_point:
                break
            else:
                print("Please select both Start and End points first.")
        elif key == 27: # ESC
            exit()

    # 3. Solve
    print("Solving maze... (This may take a second for large images)")
    raw_path = solve_bfs(img, start_point, end_point)

    if raw_path:
        print(f"Path found! Length: {len(raw_path)} pixels")
        
        # 4. Simplify for Robot
        robot_path = simplify_path(raw_path, step=ROBOT_WAYPOINT_STEP)
        print(f"Robot Waypoints: {len(robot_path)}")
        
        # 5. Visualization
        # Draw raw path (Blue thin line)
        for p in raw_path:
            img_display[p[1], p[0]] = (255, 0, 0)
            
        # Draw robot waypoints (Yellow circles) and connect them
        for i in range(len(robot_path) - 1):
            pt1 = robot_path[i]
            pt2 = robot_path[i+1]
            cv2.line(img_display, pt1, pt2, (0, 255, 255), 2)
            cv2.circle(img_display, pt1, 3, (0, 165, 255), -1)

        cv2.imshow("Solution", img_display)
        
        # 6. Output for Robot Arm
        # You would send these coordinates to your Inverse Kinematics solver
        print("\n--- ROBOT COORDINATES (Pixels) ---")
        print(robot_path)
        
        cv2.waitKey(0)
    else:
        print("No path found! Are the start and end connected by white pixels?")

    cv2.destroyAllWindows()