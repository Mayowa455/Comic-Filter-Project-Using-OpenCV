import cv2
import numpy as np

def apply_comic_filter(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return
    
    # Resize image to improve processing speed and maintain aspect ratio
    height, width = img.shape[:2]
    aspect_ratio = width / height
    new_height = 600
    new_width = int(new_height * aspect_ratio)
    img = cv2.resize(img, (new_width, new_height))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur
    gray = cv2.medianBlur(gray, 7)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # Apply bilateral filter to retain edges
    color = cv2.bilateralFilter(img, 9, 300, 300)

    # Combine edges and color
    comic = cv2.bitwise_and(color, color, mask=edges)

    # Save and display the output image
    cv2.imwrite(output_path, comic)
    cv2.imshow('Comic Filter', comic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    apply_comic_filter("input_image.jpg", "comic_output.jpg")