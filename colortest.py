import cv2
import numpy as np

def lighting_compensation(image):
    """
    Perform lighting compensation using the 'reference white' method.
    """
    image = image.astype(np.float32)
    
    # Compute luma (gamma-corrected luminance) using Rec. 709 formula
    luma = 0.2126 * image[:, :, 2] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 0]
    
    # Determine the threshold for the top 5% brightest pixels
    threshold = np.percentile(luma, 95)
    
    # Create a mask for reference white pixels
    ref_white_mask = luma >= threshold
    
    # Count the number of reference white pixels
    num_white_pixels = np.sum(ref_white_mask)

    # If not enough reference white pixels, return the original image
    if num_white_pixels <= 100:
        return image.astype(np.uint8)

    # Compute the average R, G, B values of the reference white pixels
    R_avg = np.mean(image[:, :, 2][ref_white_mask])
    G_avg = np.mean(image[:, :, 1][ref_white_mask])
    B_avg = np.mean(image[:, :, 0][ref_white_mask])

    # Normalize R, G, B channels so the reference white scales to 255
    avg_gray = (R_avg + G_avg + B_avg) / 3
    scale_R, scale_G, scale_B = 255 * (R_avg / avg_gray), 255 * (G_avg / avg_gray), 255 * (B_avg / avg_gray)

    # Apply scaling and clip to 0-255 range
    image[:, :, 2] = np.clip(image[:, :, 2] * (255 / scale_R), 0, 255)
    image[:, :, 1] = np.clip(image[:, :, 1] * (255 / scale_G), 0, 255)
    image[:, :, 0] = np.clip(image[:, :, 0] * (255 / scale_B), 0, 255)

    return image.astype(np.uint8)

def apply_edge_detection(image):
    """
    Apply edge detection (Sobel + Canny) to generate image (c).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Canny edge detection
    return edges

def apply_skin_segmentation(image):
    """
    Apply skin-tone segmentation + thresholding for binarization (d).
    """
    # Convert to YCrCb color space and extract Cr channel
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)

    # Apply threshold to detect skin regions
    _, binary = cv2.threshold(cr, 140, 255, cv2.THRESH_BINARY)
    
    return binary

def apply_skin_segmentation_hsv(image):
    """
    Apply skin-tone segmentation using HSV color space.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    return skin_mask

def apply_light_and_skin_detection(image):
    corrected_image = lighting_compensation(image)
    return apply_skin_segmentation_hsv(corrected_image)

def apply_light_and_edge_detection(image):
    corrected_image = lighting_compensation(image)
    return apply_edge_detection(corrected_image)

# Load the input image
image = cv2.imread('asl_dataset\d\hand1_d_bot_seg_1_cropped.jpeg')

# Step 1: Apply Lighting Compensation
corrected_image = lighting_compensation(image)

# Step 2: Apply Edge Detection (c)
edges = apply_edge_detection(corrected_image)

# Step 3: Apply Skin-Tone Segmentation (d)
skin_mask = apply_skin_segmentation_hsv(corrected_image)

# Save outputs
cv2.imwrite('corrected_image.jpg', corrected_image)  # (b)
cv2.imwrite('edges.jpg', edges)  # (c)
cv2.imwrite('skin_mask.jpg', skin_mask)  # (d)

# TODO: Take skin detection, apply edges in black and only in the areas where skin is detected to remove background noise
