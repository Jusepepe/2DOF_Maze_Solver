import cv2
import numpy as np
from skimage.morphology import skeletonize

def process_pipeline(image):
    """
    Standard pipeline: Gray -> Threshold -> Skeleton
    
    """
    # 1. Grayscale & Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)

    binary = ((blurred > 120)*255).astype(np.uint8)
    
    # Clean Noise
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    skeleton = skeletonize(binary > 0)
    skeleton_vis = (skeleton * 255).astype(np.uint8)
    
    return blurred, skeleton_vis