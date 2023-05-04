#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/NolanTrem/TCGA-CNN/blob/main/CompGenPythonPipelines.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Trascriptome data 

# In[1]:


#!unzip /content/drive/MyDrive/CompGenomics/transcriptome/transcriptome.zip


# In[2]:


import os
import io
import gzip
import pandas as pd
import numpy as np

dfs = {}

for filename in os.listdir():
  if filename != 'sample_data' and filename != 'transcriptome.zip' and filename != '.config' and filename != 'drive' and filename != 'merged_df.txt':
    for insidefile in os.listdir(filename):
      if insidefile != 'logs':
        filepath = os.path.join(filename, insidefile)
        with gzip.open(filepath, 'rb') as f:
            file_content = f.read().decode('utf-8')
            # Extract the cell type from the file name
            cell_type = insidefile[:-15]
            # Read in the data from the file as a DataFrame
            df = pd.read_csv(io.StringIO(file_content), sep='\t', index_col=0, names=[cell_type])
            # Add the DataFrame to the dictionary
            dfs[cell_type] = df

# Concatenate the DataFrames along the columns (axis=1)
combined_df = pd.concat(dfs.values(), axis=1)


# Remove outliers

# In[3]:


merged_df = pd.read_csv('merged_df.txt', sep='\t', header=0)


# In[4]:


combined_df = combined_df.T

gene_id = combined_df.columns
cells = combined_df.index
labels = []

for cell in cells:
  label = merged_df[merged_df['transcriptome_filename'] == (cell+'.FPKM-UQ.txt.gz')].tumor.values[0]
  if label == 'Primary Tumor':
    labels.append(1)
  elif label == 'Solid Tissue Normal':
    labels.append(0)
  else:
    labels.append(-1)

combined_df['label'] = labels

combined_df = combined_df.drop(combined_df[combined_df['label'] == -1].index)

combined_df.head()


# In[5]:


from sklearn.model_selection import train_test_split

X = combined_df.drop('label', axis=1)
y = combined_df['label']

# Normalize expression
# total_fragments = X.sum(axis=0)
# scaling_factor = total_fragments / 1e6
# X_norm = X.div(scaling_factor+0.0001, axis=1)
# X_norm = np.log2(X_norm + 1)

X_dev, X_test, y_dev, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=42)

#X_dev_stand = ss.fit_transform(X_dev)
#X_test_stand = ss.transform(X_test)


# In[61]:


y.value_counts()


# # Simple feed forward neural network

# In[43]:


from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

fpkm_scaler = Normalizer(norm='l2').fit(X_dev)
X_dev_fpkm = fpkm_scaler.transform(X_dev)
X_test_fpkm = fpkm_scaler.transform(X_test)


# In[57]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from imblearn.over_sampling import SMOTE

# Define output shape
num_classes = 2

# Define input shape
input_shape = (X_dev_fpkm.shape[1],)

# Define the model
ffnn = Sequential()

# Add first hidden layer with 64 neurons and ReLU activation
#ffnn.add(Dense(128, activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))

# Add second hidden layer with 32 neurons and ReLU activation
#ffnn.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

# Add output layer with 10 neurons and softmax activation
#ffnn.add(Dense(1, activation='sigmoid'))
smote = SMOTE(random_state=42)
X_dev_smote, y_dev_smote = smote.fit_resample(X_dev_fpkm, y_dev)
ffnn.add(Dense(7000, activation='relu', input_shape=input_shape))
ffnn.add(Dropout(0.2))
ffnn.add(Dense(7000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
ffnn.add(Dense(1, activation='sigmoid'))

ffnn.summary()


# In[58]:


from keras.optimizers import Adam

# Compile the model with categorical cross-entropy loss and Adam optimizer
ffnn.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
history_ffnn = ffnn.fit(X_dev_smote, y_dev_smote, batch_size=32, epochs=15, validation_split=0.2)


# In[59]:


import matplotlib.pyplot as plt
# a. train vs validation loss over each epoch
plt.plot(history_ffnn.history['loss'], label='Train Loss')
plt.plot(history_ffnn.history['val_loss'], label='Val Loss')
plt.title('Train vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# b. train vs validation accuracy over each epoch
plt.plot(history_ffnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_ffnn.history['val_accuracy'], label='Val Accuracy')
plt.title('Train vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[60]:


test_loss, test_accuracy = ffnn.evaluate(X_test_fpkm, y_test)
print("\n=====Results=====")
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)
#0.9795918367346939
#Test Loss:  84987.5625
#Test Accuracy:  0.9795918464660645


# In[62]:


print("Train loss: ", history_ffnn.history['loss'])
print("Val loss: ", history_ffnn.history['val_loss'])
print("Train acc: ", history_ffnn.history['accuracy'])
print("Val acc: ", history_ffnn.history['val_accuracy'])


# In[35]:


from sklearn.metrics import confusion_matrix
y_pred = ffnn.predict(X_test)
confusion_matrix(y_test, y_pred)


# # SCOPE

# In[5]:


get_ipython().system('pip install scikeras')


# In[6]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.ensemble import VotingClassifier


# In[7]:


# Preprocess data
fpkm_scaler = Normalizer().fit(X_dev)
X_train_rpkm = fpkm_scaler.transform(X_dev)
X_test_rpkm = fpkm_scaler.transform(X_test)


# In[8]:


import scikeras
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier

# Define models
def create_model_None17k():
    nn = Sequential()
    nn.add(Dense(7000, activation='relu', input_shape=(X_dev.shape[1],)))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return nn

def create_model_None17kDropout():
    nn = Sequential()
    nn.add(Dense(7000, activation='relu', input_shape=(X_dev.shape[1],)))
    nn.add(Dropout(0.1))
    nn.add(Dense(7000, activation='relu'))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return nn

def create_model_SmoteNone17k():
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_dev, y_dev)
    nn = Sequential()
    nn.add(Dense(7000, activation='relu', input_shape=(X_train_smote.shape[1],)))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return nn

def create_model_Rm500():
    rank_scaler = Normalizer().fit(X_dev)
    X_train_rank = rank_scaler.transform(X_dev)
    X_test_rank = rank_scaler.transform(X_test)
    minmax_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train_rank)
    X_train_scaled = minmax_scaler.transform(X_train_rank)
    X_test_scaled = minmax_scaler.transform(X_test_rank)
    nn = Sequential()
    nn.add(Dense(500, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return nn

def create_model_Rm500Dropout():
    rank_scaler = Normalizer().fit(X_dev)
    X_train_rank = rank_scaler.transform(X_dev)
    X_test_rank = rank_scaler.transform(X_test)
    minmax_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train_rank)
    X_train_scaled = minmax_scaler.transform(X_train_rank)
    X_test_scaled = minmax_scaler.transform(X_test_rank)
    nn = Sequential()
    nn.add(Dense(500, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    nn.add(Dropout(0.1))
    nn.add(Dense(500, activation='relu'))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return nn

# Wrap models into a voting classifier
estimators = [
    ('None17k', KerasClassifier(build_fn=create_model_None17k, epochs=10, batch_size=32, verbose=1)),
    ('None17kDropout', KerasClassifier(build_fn=create_model_None17kDropout, epochs=10, batch_size=32, verbose=1)),
    ('SmoteNone17k', KerasClassifier(build_fn=create_model_SmoteNone17k, epochs=10, batch_size=32, verbose=1)),
    ('Rm500', KerasClassifier(build_fn=create_model_Rm500, epochs=10, batch_size=32, verbose=1)),
    ('Rm500Dropout', KerasClassifier(build_fn=create_model_Rm500Dropout, epochs=10, batch_size=32, verbose=1))
]
ensemble = VotingClassifier(estimators)

# Fit and evaluate the ensemble classifier on the data
history_ensemble = ensemble.fit(X_dev, y_dev)

# Evaluate the ensemble classifier on the test data
print("accuracy: ", ensemble.score(X_test, y_test))


# In[9]:


from sklearn.metrics import log_loss, accuracy_score

y_pred = ensemble.predict(X_test)
test_loss = log_loss(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Loss: ", test_loss)


# In[13]:


from sklearn.metrics import log_loss, accuracy_score


y_pred = ensemble.predict(X_test)
test_loss = log_loss(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Loss: ", test_loss)


# In[11]:


from sklearn.metrics import confusion_matrix
y_pred = ensemble.predict(X_test)
confusion_matrix(y_test, y_pred)


# # WSI Pipeline

# In[1]:


import os
import cv2

datasetPath = "/content/drive/MyDrive/Computational Genomics Project"
solidTissueNormal = []
primaryTumor = []

for filename in os.listdir(datasetPath):
  if filename[-12:] == "Cancerimages" and filename[:5] != "brain":
    path = os.path.join(datasetPath, filename, "resizedImages", "solidTissueNormal")
    for img_file in os.listdir(path):
      img = cv2.imread(os.path.join(path, img_file))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      solidTissueNormal.append(img)
    path = os.path.join(datasetPath, filename, "resizedImages", "primaryTumor")
    for img_file in os.listdir(path):
      img = cv2.imread(os.path.join(path, img_file))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      primaryTumor.append(img)


# In[29]:


import random
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Create label
images = solidTissueNormal.copy()
images.extend(primaryTumor)
labels = [0] * len(solidTissueNormal)
labels.extend([1] * len(primaryTumor))

# Shuffle order
temp = list(zip(images, labels))
random.shuffle(temp)
X, y = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
X, y = np.array(X), np.array(y)

# Split data
X_dev, X_test, y_dev, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, stratify=y_dev, test_size=0.1, random_state=42)


# In[5]:


unique_values, counts = np.unique(y, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"Class {value}: {count} instances")


# Augment

# In[30]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2
    #channel_shift_range=20,  # adjust channel shift
    #brightness_range=(0.8, 1.2)  # adjust brightness
    )

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20

train_generator = train_datagen.flow(X_train, y_train, batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size)

X_test_final = test_datagen.flow(X_test, batch_size=len(X_test), shuffle=False).next()


# In[17]:


import tensorflow as tf

input_shape = (500,500,3)
#X_val_images = X_val_images * 1. / 255

#X_train_images = tf.image.per_image_standardization(X_train_final)
#X_val_images = tf.image.per_image_standardization(X_val)
#X_test_images = tf.image.per_image_standardization(X_test)

#X_dev_images = X_dev.reshape(X_dev.shape[0], 227, 227, 3)
#X_test_images = X_test.reshape(X_test.shape[0], 227, 227, 3)
num_classes = 2


# python version of alexnet

# In[31]:


from tensorflow import keras
import matplotlib.pyplot as plt
import time

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])

model.summary()


# In[41]:


from keras.optimizers import Adam
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=1e-5), metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32, # number of batches per epoch
    epochs=20,
    validation_data=val_generator,
    validation_steps=len(X_val) // 32 # number of batches for validation
)


# In[42]:


import matplotlib.pyplot as plt
# a. train vs validation loss over each epoch
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Train vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# b. train vs validation accuracy over each epoch
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Train vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[43]:


test_loss, test_accuracy = model.evaluate(X_test_final, y_test)
print("\n=====Results=====")
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)


# In[58]:


X_test_final.shape


# In[59]:


print("Train loss: ", history.history['loss'])
print("Val loss: ", history.history['val_loss'])
print("Train acc: ", history.history['accuracy'])
print("Val acc: ", history.history['val_accuracy'])


# In[30]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test_images)
print(y_pred)

