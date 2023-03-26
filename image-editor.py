import os
import cv2 as cv
import numpy as np

# Unedited images location
directory = '/home/janamejay/Documents/Image-Classification/images'

edited_img_dir = '/home/janamejay/Documents/Image-Classification/edited-images'

for filename in os.listdir(directory):
    image_path = os.path.join(directory, filename)
    image = cv.imread(image_path)
    print(f"Found {filename}")

    # Convert BGR to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Convert to grayscale
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # Resize to 32x32 pixels
    image = cv.resize(image, (32,32))
    print(f"Converted to grayscale and resized to 32x32 pixels.")
    
    # Save edited image
    os.chdir(edited_img_dir)
    cv.imwrite(filename, image)
