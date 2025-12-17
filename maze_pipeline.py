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
    Sorts points into specific order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    This allows the user to click corners in any order.
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Top-Left has the smallest sum(x+y)
    # Bottom-Right has the largest sum(x+y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-Right has the smallest difference(y-x)
    # Bottom-Left has the largest difference(y-x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Threshold
    # Adjust Block Size (11) and C (2) based on your lighting
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Clean Noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 4. Skeletonize
    # Convert to boolean (0/1) for skimage, then back to uint8 (0-255) for OpenCV
    skeleton = skeletonize(binary > 0)
    skeleton_vis = (skeleton * 255).astype(np.uint8)
    
    return binary, skeleton_vis

# --- MAIN LOOP ---
if __name__ == "__main__":
    # LOAD IMAGE
    # Replace with your actual image path or use 0 for webcam (see note below)
    img = cv2.imread('maze_photo.webp') 

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
    binary_map, skeleton_map = process_pipeline(warped_view)

    # 4. Show Results
    cv2.imshow("1. Warped View", warped_view)
    cv2.imshow("2. Binary Map", binary_map)
    cv2.imshow("3. Skeleton (Path Center)", skeleton_map)

    print("Processing complete. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()