import cv2
import numpy as np
import os

def detect_hand_edges(image_path, output_path=None, debug=False):
    """
    Detects the hand in an image, applies edge detection only to the hand region,
    and returns a processed image with the hand edges highlighted.
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the processed image. If None, doesn't save.
        debug (bool): If True, saves intermediate images for debugging
        
    Returns:
        numpy.ndarray: Processed image with hand edges highlighted
    """
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (244, 244))
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return None
    
    debug_dir = os.path.dirname(output_path) if output_path else "."
    
    # Convert to YCrCb color space
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Skin detection in YCrCb color space
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "1_skin_mask.jpg"), skin_mask)
    
    # Find contours and select the largest one (assumed to be the hand)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_mask = np.zeros_like(skin_mask)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:  # Threshold to ignore small objects
            cv2.drawContours(hand_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "2_hand_mask.jpg"), hand_mask)
    
    # Apply the hand mask to isolate the hand region
    hand_region = cv2.bitwise_and(img, img, mask=hand_mask)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "3_hand_region.jpg"), hand_region)
    
    # Convert the hand region to grayscale
    gray_hand = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray_hand, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "4_hand_edges.jpg"), edges)
    
    # Ensure edges are only within the detected hand
    final_edges = cv2.bitwise_and(edges, edges, mask=hand_mask)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "5_final_hand_edges.jpg"), final_edges)
    
    return final_edges
