# Augmentation
Apply data augmentation techniques to images using the Keras library.
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load an example image
image = cv2.imread('input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create an ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate augmented images
img = np.expand_dims(image, axis=0)
aug_iter = datagen.flow(img)

# Display augmented images
fig, ax = plt.subplots(1, 4, figsize=(15, 5))
for i in range(4):
    augmented_image = next(aug_iter)[0].astype(np.uint8)
    ax[i].imshow(augmented_image)
    ax[i].axis('off')

plt.show()
