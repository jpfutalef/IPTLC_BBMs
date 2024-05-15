from pathlib import Path
import torch

# dataset_location = Path("data/IO-datasets/TLCN/2024-05-10_13-10-13/dataset_array_bottom_up.pt") # MCS
dataset_location = Path("data/IO-datasets/TLCN/2024-05-10_17-58-14/dataset_array_bottom_up.pt")  # RESTART
dataset = torch.load(dataset_location)

# %% Classifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import datasets, neighbors, linear_model, tree, metrics
import xgboost as xgb

import numpy as np
import sys
import matplotlib.pyplot as plt
import time

# %% Setup input and output. Turn tensors into numpy arrays
X = [x[0].numpy() for x in dataset]
y = [x[1].numpy() for x in dataset]

X = np.array(X)
# y = np.array(y)
y = np.array(y)[:, [0, 4, 5]]

print(f"Full input size: {X.shape}")

# %% Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Training input size: {X_train.shape}")
print(f"Test input size: {X_test.shape}")

# %% Train the classifier
model = xgb.XGBClassifier(max_depth=3, n_estimators=5)
model.fit(X_train, y_train)

# Training accuracy
y_pred_train = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
print("Training Accuracy: %.5f%%" % (accuracy * 100.0))

# Test accuracy
y_pred_train = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_train)
print("Accuracy: %.5f%%" % (accuracy * 100.0))

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

# %% Count zeros in the columns of the outputs
print("Zeros in y")
print(sum(np.where(y == 0, 1, 0)))

print("Zeros in y_train")
print(sum(np.where(y_train == 0, 1, 0)))

print("Zeros in y_test")
print(sum(np.where(y_test == 0, 1, 0)))
