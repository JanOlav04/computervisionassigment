import cv2
import numpy as np


def rgb_to_ycbcr(image_path):
    # Loading image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))

    # Converting image to YCbCr
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Define skin color range in YCbCr
    lower_skin = np.array([0, 133, 77], dtype = np.uint8)
    upper_skin = np.array([255, 173, 127], dtype = np.uint8)

    # Create skin mask
    mask = cv2.inRange(img_ycbcr, lower_skin, upper_skin)

    # Apply mask to original image
    result = cv2.bitwise_and(img, img, mask = mask)

    # Display images
    cv2.imshow("YCbCr Skin Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result

rgb_to_ycbcr("sample.jpg")
