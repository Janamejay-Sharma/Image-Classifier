import os
import cv2 as cv
import numpy as np
from tensorflow import keras
from keras import models

# List of classes in CIFAR10 image database
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Address of edited images directory
edited_img_dir = '/home/janamejay/Documents/Image-Classification/edited-images'

model = models.load_model('/home/janamejay/Documents/Image-Classification/image-classifier.model')

for filename in os.listdir(edited_img_dir):
    image_path = os.path.join(edited_img_dir, filename)
    image = cv.imread(image_path)

    # List of probabilities of each image class
    prediction = model.predict(np.array([image]))
    
    # Index with highest probability is predicted class
    predicted_class = class_names[np.argmax(prediction)]

    print(f"Image: {filename}, prediction: {predicted_class}")

    