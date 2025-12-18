import cv2
import numpy as np
from skimage.morphology import skeletonize

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

def process_pipeline(image):
    """
    Standard pipeline: Gray -> Threshold -> Skeleton
    """
    # 1. Grayscale & Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)

    binary = ((blurred > 100)*255).astype(np.uint8)
    
    # Clean Noise
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    skeleton = skeletonize(binary > 0)
    skeleton_vis = (skeleton * 255).astype(np.uint8)
    
    return blurred, skeleton_vis

# --- MAIN LOOP ---
if __name__ == "__main__":
    # LOAD IMAGE
    # Replace with your actual image path or use 0 for webcam (see note below)
    img = cv2.imread('maze_photo4.jpeg') 

    if img is None:
        print("Error: Image not found.")
        exit()

    # Create a copy for the user to interact with (so we don't warp the red dots later)
    img_display = img.copy()

    print("--- INSTRUCTIONS ---")
    print("1. Click the 4 corners of the maze in the image window.")
    print("2. Order does not matter (the script will sort them).")
    print("3. Press any key after selecting 4 points to process.")

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
    warped_view = warp_perspective(img, sorted_pts)
    
    # 3. Vision Pipeline
    blurred, binary_map = process_pipeline(warped_view)

    # 4. Show Results
    cv2.imshow("1. Warped View", warped_view)
    cv2.imshow("1. Blurred", blurred)
    cv2.imshow("2. Binary Map", binary_map)

    print("Processing complete. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()