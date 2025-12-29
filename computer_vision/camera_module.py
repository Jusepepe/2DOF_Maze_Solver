import cv2
import numpy as np

# --- GLOBAL VARIABLES ---
corners = []      # Stores the 4 clicked points
img_display = None # Image used for UI interaction

def mouse_callback(event, x, y, flags, param):
    """
    Records mouse clicks and draws visual feedback.

    """
    global corners, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only accept clicks if we haven't selected 4 points yet
        if len(corners) < 4:
            corners.append((x, y))
            
            # Visual Feedback: Draw a red circle and number the point
            cv2.circle(img_display, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(img_display, str(len(corners)), (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Select 4 Corners", img_display)
            print(f"Point {len(corners)}: ({x}, {y})")

def order_points(pts):
    """
    Robustly orders points as: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    # 1. Convert to a list of lists/tuples just in case
    pts = [list(x) for x in pts]
    
    # 2. Sort all points by their Y-coordinate (Ascending)
    # The top two points will have the smallest Y values
    pts.sort(key=lambda x: x[1])
    
    # Slice the top 2 (Top-Left and Top-Right) and bottom 2 (Bottom-Left and Bottom-Right)
    top_most = pts[:2]
    bottom_most = pts[2:]
    
    # 3. Sort the top points by their X-coordinate
    # Smallest X is Top-Left, Largest X is Top-Right
    top_most.sort(key=lambda x: x[0])
    tl = top_most[0]
    tr = top_most[1]
    
    # 4. Sort the bottom points by their X-coordinate
    # Smallest X is Bottom-Left, Largest X is Bottom-Right
    bottom_most.sort(key=lambda x: x[0])
    bl = bottom_most[0]
    br = bottom_most[1]
    
    # 5. Return in the order expected by warp_perspective: TL, TR, BR, BL
    return np.array([tl, tr, br, bl], dtype="float32")

def warp_perspective(image, sorted_corners, output_size=(600, 600)):
    """
    Warps the region defined by sorted_corners into a flat top-down view.
    """
    width, height = output_size
    dst_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])
    
    matrix = cv2.getPerspectiveTransform(sorted_corners, dst_points)
    warped_img = cv2.warpPerspective(image, matrix, output_size)
    return warped_img

def start_camera_viewer(source=0, window_name="Select 4 Corners"):
    """
    Opens a video source, displays frames, and uses mouse_callback to collect clicks.
    Press 'q' or ESC to quit.
    """
    global img_display, corners
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: could not open camera source:", source)
        return

    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show frame and redraw any previously clicked points
        img_display = frame.copy()
        
        print("4 points collected. Press any key to process...")
        cv2.waitKey(0) # Wait indefinitely for a key press
        break

    cap.release()
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    # Quick manual test: open default camera
    start_camera_viewer(source=0)

    # Setup Window and Callback
    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", mouse_callback)
    cv2.imshow("Select 4 Corners", img_display)

    # Wait loop
    while True:
        key = cv2.waitKey(10) & 0xFF
        
        # If 4 points selected, wait for user to hit a key to confirm
        if len(corners) == 4:
            print("4 points collected. Press any key to process...")
            cv2.waitKey(0) # Wait indefinitely for a key press
            break
        
        # 'q' to quit early
        if key == ord('q'):
            exit()

    # --- PROCESSING ---
    cv2.destroyWindow("Select 4 Corners")
    
    # 1. Sort Points
    sorted_pts = order_points(corners)
    
    # 2. Warp
    # Note: We pass 'img' (clean), not 'img_display' (which has red dots)
    warped_view = warp_perspective(img_display, sorted_pts)

    # 4. Show Results
    cv2.imshow("1. Warped View", warped_view)

    print("Processing complete. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()