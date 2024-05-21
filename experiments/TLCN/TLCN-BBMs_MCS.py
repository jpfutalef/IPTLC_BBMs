from pathlib import Path
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import datasets, neighbors, linear_model, tree, metrics
import xgboost as xgb

import numpy as np
import sys
import matplotlib.pyplot as plt
import time

#%% Locations and data loading
dataset_location = Path("data/IO-datasets/TLCN/MCS_data/2024-05-16_11-55-12/dataset_array_bottom_up.pt") # MCS
# dataset_location = Path("data/IO-datasets/TLCN/2024-05-15_18-38-44/dataset_array_bottom_up.pt")  # RESTART

# %% Try opening the dataset. If it fails, try loading it as a numpy array
try:
    print("Attempting to load the dataset as a numpy array...")
    X_loc = dataset_location.parent / "np_array_X.npy"
    y_loc = dataset_location.parent / "np_array_y.npy"

    X = np.load(str(X_loc))
    y = np.load(str(y_loc))

except FileNotFoundError:
    print("Not found. Loading the dataset as a torch array...")
    dataset = torch.load(dataset_location)
    X = [x[0].numpy() for x in dataset]
    y = [x[1].numpy() for x in dataset]

    X = np.array(X)
    y = np.array(y)
    # y = np.array(y)[:, [0, 4, 5]]

    # Save as numpy arrays
    print(f"Saving as numpy arrays in {dataset_location.parent}")
    np.save(str(dataset_location.parent / "np_array_X.npy"), X)
    np.save(str(dataset_location.parent / "np_array_y.npy"), y)

print(f"Full input size: {X.shape}")

# %% Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Training input size: {X_train.shape}")
print(f"Test input size: {X_test.shape}")

# %% Count zeros in the columns of the outputs
print("Zeros in Y columns")
print(sum(np.where(y == 0, 1, 0)))

# %% Train the classifier 1
model = xgb.XGBClassifier(max_depth=5, n_estimators=3)
model.fit(X_train, y_train)

# Training accuracy
y_pred_train = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
print("Training Accuracy: %.5f%%" % (accuracy * 100.0))

# Test accuracy
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy: %.5f%%" % (accuracy * 100.0))

# Training confusion matrix for each output
for i in range(y_train.shape[1]):
    cm = confusion_matrix(y_train[:, i], y_pred_train[:, i], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Training - Classifier 1 - Output %d" % i)
    plt.show()

# Test confusion matrix for each output
for i in range(y_train.shape[1]):
    cm = confusion_matrix(y_test[:, i], y_pred_test[:, i], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Test - Classifier 1 - Output %d" % i)
    plt.show()

# %% Train the classifier 2
model = xgb.XGBClassifier(max_depth=10, n_estimators=3)
model.fit(X_train, y_train)

# Training accuracy
y_pred_train = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
print("Training Accuracy: %.5f%%" % (accuracy * 100.0))

# Test accuracy
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy: %.5f%%" % (accuracy * 100.0))

# Training confusion matrix for each output
for i in range(y_train.shape[1]):
    cm = confusion_matrix(y_train[:, i], y_pred_train[:, i], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Training - Classifier 2 - Output %d" % i)
    plt.show()

# Test confusion matrix for each output
for i in range(y_train.shape[1]):
    cm = confusion_matrix(y_test[:, i], y_pred_test[:, i], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Test - Classifier 2 - Output %d" % i)
    plt.show()

# %% Train the classifier 3
model = xgb.XGBClassifier(max_depth=5, n_estimators=8)
model.fit(X_train, y_train)

# Training accuracy
y_pred_train = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
print("Training Accuracy: %.5f%%" % (accuracy * 100.0))

# Test accuracy
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy: %.5f%%" % (accuracy * 100.0))

# Training confusion matrix for each output
for i in range(y_train.shape[1]):
    cm = confusion_matrix(y_train[:, i], y_pred_train[:, i], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Training - Classifier 3 - Output %d" % i)
    plt.show()

# Test confusion matrix for each output
for i in range(y_train.shape[1]):
    cm = confusion_matrix(y_test[:, i], y_pred_test[:, i], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Test - Classifier 3 - Output %d" % i)
    plt.show()

# %% Train the classifier 4
model = xgb.XGBClassifier(max_depth=10, n_estimators=8)
model.fit(X_train, y_train)

# Training accuracy
y_pred_train = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
print("Training Accuracy: %.5f%%" % (accuracy * 100.0))

# Test accuracy
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy: %.5f%%" % (accuracy * 100.0))

# Training confusion matrix for each output
for i in range(y_train.shape[1]):
    cm = confusion_matrix(y_train[:, i], y_pred_train[:, i], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Training - Classifier 4 - Output %d" % i)
    plt.show()

# Test confusion matrix for each output
for i in range(y_train.shape[1]):
    cm = confusion_matrix(y_test[:, i], y_pred_test[:, i], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Test - Classifier 4 - Output %d" % i)
    plt.show()
