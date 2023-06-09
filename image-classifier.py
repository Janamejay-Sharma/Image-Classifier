from tensorflow import keras
from keras import datasets, models, layers

"""
    The function call datasets.cifar10.load_data() returns:
        (x_train, y_train), (x_test, y_test), where:
    x_train: uint8 NumPy array of grayscale image data with shapes (50000, 32, 32, 3), containing the training data. Pixel values range from 0 to 255.
    y_train: uint8 NumPy array of labels (integers in range 0-9) with shape (50000, 1) for the training data.
    x_test: uint8 NumPy array of grayscale image data with shapes (10000, 32, 32, 3), containing the test data. Pixel values range from 0 to 255.
    y_test: uint8 NumPy array of labels (integers in range 0-9) with shape (10000, 1) for the test data.
"""
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Reduce RGB values to a scale of 0 to 1, for faster training
training_images = training_images / 255
testing_images = testing_images / 255

# List of classes in CIFAR10 image database
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Select first 20,000 training images
training_images = training_images[:20000]
training_labels = training_labels[:20000]

# Select first 4000 testing images
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Define the neural network
model = models.Sequential()

"""
    Add a convolution layer with 32 neurons (filters, or feature maps)
    Type: rectified linear unit (ReLU)
    Input: 32x32 pixel image with 3 color channels 
""" 
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)))


"""
    Downsample the input along its height and width,
    by taking the maximum value over an input window (of pool_size 2x2), for each channel of the input.
    
    Achieved by:
      dividing the input image or feature map into non-overlapping rectangular regions,
      and taking the maximum value within each region to produce a new, smaller feature map.
"""
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Add another convolution layer with 64 feature maps
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Add another convolution layer with 64 feature maps
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

# Flatten input to one dimension
model.add(layers.Flatten())

# Performs the operation: output = activation(dot(input, kernel) + bias)
model.add(layers.Dense(units=64, activation='relu'))

"""
    Output layer, 10 units, 1 for each class of image
    'softmax' normalizes output so all values sum to 1
    This is useful since we want the probability of an image belonging to a class
"""
model.add(layers.Dense(units=10, activation='softmax'))

"""
    Loss function measures error or deviation
    "sparse" allows only non-zero values to be added to the matrix of labels
    "categorical" as the result is not binary (10 possibilities)
"""
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=training_images, y=training_labels, batch_size=1, epochs=10, validation_data=(testing_images, testing_labels))

# Store and print results
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the trained model
model.save('/home/janamejay/Documents/Image-Classification/image-classifier.model')
