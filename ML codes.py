# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:25:33 2023

@author: CC
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import imgaug.augmenters as iaa
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Step 1: Data collection
# Assuming you have separate folders for train, test, and val
train_folder = r"C:/Users/CC/Downloads/Pneumonia/train"
test_folder = r"C:/Users/CC/Downloads/Pneumonia/test"
val_folder = r"C:/Users/CC/Downloads/Pneumonia/val"

import os

# Counting the number of images in each folder
train_count = len(os.listdir(train_folder + '/PNEUMONIA')) + len(os.listdir(train_folder + '/NORMAL'))
test_count = len(os.listdir(test_folder + '/PNEUMONIA')) + len(os.listdir(test_folder + '/NORMAL'))
val_count = len(os.listdir(val_folder + '/PNEUMONIA')) + len(os.listdir(val_folder + '/NORMAL'))

# Plotting the data distribution
labels = ['Training Set', 'Testing Set', 'Validation Set']
counts = [train_count, test_count, val_count]

plt.bar(labels, counts)
plt.xlabel('Dataset')
plt.ylabel('Number of Images')
plt.title('Data Distribution')
plt.show()



train_pneumonia_files = glob.glob(train_folder + '/PNEUMONIA/*.jpeg')
train_normal_files = glob.glob(train_folder + '/NORMAL/*.jpeg')

test_pneumonia_files = glob.glob(test_folder + '/PNEUMONIA/*.jpeg')
test_normal_files = glob.glob(test_folder + '/NORMAL/*.jpeg')

val_pneumonia_files = glob.glob(val_folder + '/PNEUMONIA/*.jpeg')
val_normal_files = glob.glob(val_folder + '/NORMAL/*.jpeg')

# Create a DataFrame to store the file paths and labels
train_data = pd.DataFrame({'image_path': train_pneumonia_files + train_normal_files, 'label': ['pneumonia'] * len(train_pneumonia_files) + ['normal'] * len(train_normal_files)})
test_data = pd.DataFrame({'image_path': test_pneumonia_files + test_normal_files, 'label': ['pneumonia'] * len(test_pneumonia_files) + ['normal'] * len(test_normal_files)})
val_data = pd.DataFrame({'image_path': val_pneumonia_files + val_normal_files, 'label': ['pneumonia'] * len(val_pneumonia_files) + ['normal'] * len(val_normal_files)})

# EDA: Visualize Sample Images
num_samples = 5

# Randomly select num_samples samples from the test set
sample_indices = np.random.choice(len(train_data), num_samples, replace=False)
sample_images = train_data.iloc[sample_indices]

# Display the sample images with labels
fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))

for i, row in enumerate(sample_images.iterrows()):
    image_path = row[1]['image_path']
    label = row[1]['label'] 
    image = Image.open(image_path)
    axes[i].imshow(image)
    axes[i].set_title(label)
    axes[i].axis('off')

plt.show()

# Step 2: Data preprocessing
# Assuming you have a function that extracts features from images
def extract_features(image_paths):
    # Placeholder code for extracting features (replace with your actual implementation)
    # This example returns random features
    return np.random.rand(len(image_paths), 10)

X_train = extract_features(train_data['image_path'])
X_test = extract_features(test_data['image_path'])
X_val = extract_features(val_data['image_path'])

# Step 3: Feature selection
# Assuming you want to select all available features
selector = SelectKBest(f_classif, k='all')
X_train = selector.fit_transform(X_train, train_data['label'])
X_test = selector.transform(X_test)
X_val = selector.transform(X_val)

# Step 4: Data augmentation
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flips with a probability of 0.5
    iaa.Crop(percent=(0, 0.1)),  # Crop images by 0-10% of their height/width
    iaa.GaussianBlur(sigma=(0, 0.5)),  # Apply Gaussian blur with a sigma of 0-0.5
    iaa.Affine(rotate=(-10, 10))  # Rotate images by -10 to 10 degrees
])

# Apply data augmentation to training data
X_train_augmented = augmenter(images=X_train)

# Step 4: Model selection and training
scaler = StandardScaler()
X_train_augmented = scaler.fit_transform(X_train_augmented)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

model1 = LogisticRegression(random_state=42)
model1.fit(X_train, train_data['label'])

model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train, train_data['label'])

model3 = RandomForestClassifier(n_estimators=100, random_state=42)  # Set the desired number of estimators
model3.fit(X_train, train_data['label'])



# Step 5: Model evaluation
y_pred = model1.predict(X_test)
accuracy_model1 = accuracy_score(test_data['label'], y_pred)

y_pred = model2.predict(X_test)
accuracy_model2 = accuracy_score(test_data['label'], y_pred)

y_pred = model3.predict(X_test)
accuracy_model3 = accuracy_score(test_data['label'], y_pred)


print("Accuracy of LogisticRegression:", accuracy_model1)
print("Accuracy of DecisionTree:", accuracy_model2)
print("Accuracy of RandomForest:", accuracy_model3)

# List of model names
models = ['Model1', 'Model2', 'Model 3']

# List of accuracies for each model
accuracies = [accuracy_model1, accuracy_model2, accuracy_model3]

# Plot the accuracies
plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.show()





