import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (Input, Conv2D, Activation, BatchNormalization,
                                     MaxPooling2D, Flatten, Dense, Dropout)
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import matplotlib.pyplot as plt

# Paths
dataset_path = '/Volumes/NolansDrive/TCGA-CNN/commonCancerDataset/augmentedImages'
validation_path = '/Volumes/NolansDrive/TCGA-CNN/commonCancerDataset/originalImages'

# Image size and batch size
image_size = (500, 500)
batch_size = 64

# Create augmented image data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    validation_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the AlexNet model
model = Sequential([
    Input(shape=(*image_size, 3), name='data'),
    Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv1'),
    BatchNormalization(),
    MaxPooling2D((3, 3), strides=(2, 2), name='pool1'),
    Conv2D(256, (5, 5), padding='same', activation='relu', name='conv2'),
    BatchNormalization(),
    MaxPooling2D((3, 3), strides=(2, 2), name='pool2'),
    Conv2D(384, (3, 3), padding='same', activation='relu', name='conv3'),
    Conv2D(384, (3, 3), padding='same', activation='relu', name='conv4'),
    Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5'),
    MaxPooling2D((3, 3), strides=(2, 2), name='pool5'),
    Flatten(),
    Dense(4096, activation='relu', name='fc6'),
    Dropout(0.5),
    Dense(4096, activation='relu', name='fc7'),
    Dropout(0.5),
    Dense(2, activation='softmax', name='fc8')
])

# Compile the model
model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Set training options
lr_reducer = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6, verbose=1)
early_stopper = EarlyStopping(min_delta=0, patience=10, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=1,
    callbacks=[lr_reducer, early_stopper]
)

# Optionally, you can save the trained model
model.save('/Volumes/NolansDrive/TCGA-CNN/commonCancerDataset/augmentedImages/alexnet_trained_nov_1.h5')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save the plots to files
plt.savefig('/Volumes/NolansDrive/TCGA-CNN/commonCancerDataset/augmentedImages/accuracy_loss_plot.png')