from pathlib import Path
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import datasets, neighbors, linear_model, tree, metrics
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

import numpy as np
import sys
import matplotlib.pyplot as plt
import time

# %% Locations and data loading
# dataset_location = Path("data/IO-datasets/TLCN/2024-05-10_13-10-13/dataset_array_bottom_up.pt") # MCS
dataset_location = Path("data/IO-datasets/TLCN/RESTART_data/2024-05-16_11-56-28/dataset_array_bottom_up.pt")  # RESTART

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
    np.save(str(dataset_location.parent / "np_array_X.npy"), X)
    np.save(str(dataset_location.parent / "np_array_y.npy"), y)

print(f"Full input size: {X.shape}")

# %% Select columns
y_columns = [5]
y = y[:, y_columns]

# %% Count zeros in the columns of the outputs
print("--- Before balancing ---")

# Count zeros in the columns of the outputs
print("Zeros in Y columns")
n_zeros = sum(np.where(y == 0, 1, 0))
print(n_zeros)

# %% Balance dataset. Many rows from [0:8] are repeated. So, we will balance the dataset by removing the repeated rows
# Get the unique rows
X_unique, indices_unique = np.unique(X, axis=0, return_index=True)

# Get the rows where there's at least one zero in the outputs
y_with_zero = np.where(np.sum(y, axis=1) != y.shape[1], True, False)

# Get the indices of the rows with zero
indices_with_zero = np.where(y_with_zero)[0]

# Merge the unique and zero indices
indices = np.concatenate((indices_unique, indices_with_zero))
indices = np.unique(indices)

# Get the unique rows with zeros
X_unique = X[indices]
y_unique = y[indices]

# %% Count zeros in the columns of the outputs
print("--- After balancing ---")

print("Zeros in y")
n_zeros = sum(np.where(y_unique == 0, 1, 0))
print(n_zeros)

# %% Set the dataset as the balanced one
X = X_unique
y = y_unique

# %% Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Training input size: {X_train.shape}")
print(f"Test input size: {X_test.shape}")


# %% A function to train the classifier
def train_classifier(X_train, y_train, X_test, y_test, model_name, model=None, **kwargs):
    # Create the model
    if model is None:
        model = xgb.XGBClassifier(**kwargs)

    # Train the model
    model.fit(X_train, y_train)

    # Training accuracy
    y_pred_train = model.predict(X_train)
    accuracy = roc_auc_score(y_train, y_pred_train)
    print(f"Training Accuracy of {model_name}: %.5f%%" % (accuracy * 100.0))

    # Test accuracy
    y_pred_test = model.predict(X_test)
    accuracy = roc_auc_score(y_test, y_pred_test)
    print("Test Accuracy: %.5f%%" % (accuracy * 100.0))

    # Training confusion matrix for each output
    cm_train_list = []
    cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
    cm_train_list.append(cm_train)

    # Test confusion matrix for each output
    cm_test_list = []
    cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    cm_test_list.append(cm_test)

    return model, cm_train_list, cm_test_list


# %%
def printA(a):
    for row in a:
        for col in row:
            print("{:5}".format(col), end=" ")
        print("")
    print("")


# %% Pyplot normal style
plt.style.use('default')

# %% Classifier 1
model_name = "Classifier 1"
model = xgb.XGBClassifier(max_depth=5,
                            n_estimators=3)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Scores
score_train = roc_auc_score(y_train, y_pred_train)
score_test = roc_auc_score(y_test, y_pred_test)

# Figures for the confusion matrices
fig, ax = plt.subplots(2, 1, figsize=(2.5, 4), constrained_layout=True)
ax = ax.flatten()

# Confusion matrix for each output
print(f"--- {model_name} (Training) ---")
print(f"Training AUC: {score_train}")
# cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
# ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(cmap="Greys", ax=ax[0], colorbar=False)
# printA(cm_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, ax=ax[0], cmap="Greys", colorbar=False,
                                        normalize="pred")
ax[0].set_title(f"Training - {model_name}")

print(f"--- {model_name} (Test) ---")
print(f"Test AUC: {score_test}")
# cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
# ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(cmap="Greys", ax=ax[1], colorbar=False)
# printA(cm_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax[1], cmap="Greys", colorbar=False,
                                        normalize="pred")
ax[1].set_title(f"Test - {model_name}")

# Show the figures
fig.show()

# %% Classifier 2
model_name = "Classifier 2"
model = xgb.XGBClassifier(max_depth=10,
                          n_estimators=3)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Scores
score_train = roc_auc_score(y_train, y_pred_train)
score_test = roc_auc_score(y_test, y_pred_test)

print(f"Training AUC: {score_train}")
print(f"Test AUC: {score_test}")

# Figures for the confusion matrices
fig, ax = plt.subplots(2, 1, figsize=(2, 4))
ax = ax.flatten()

# Confusion matrix for each output
print(f"--- {model_name} (Training) ---")
cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(cmap="Greys", ax=ax[0], colorbar=False)
ax[0].set_title(f"Training - {model_name}")
printA(cm_train)

print(f"--- {model_name} (Test) ---")
cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(cmap="Greys", ax=ax[1], colorbar=False)
ax[1].set_title(f"Test - {model_name}")
printA(cm_test)

# Show the figures
plt.tight_layout()
plt.show()

# %% Classifier 3
model_name = "Classifier 3"
model = xgb.XGBClassifier(max_depth=5,
                          n_estimators=8)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Scores
score_train = roc_auc_score(y_train, y_pred_train)
score_test = roc_auc_score(y_test, y_pred_test)

print(f"Training AUC: {score_train}")
print(f"Test AUC: {score_test}")

# Figures for the confusion matrices
fig, ax = plt.subplots(2, 1, figsize=(2, 4))
ax = ax.flatten()

# Confusion matrix for each output
print(f"--- {model_name} (Training) ---")
cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(cmap="Greys", ax=ax[0], colorbar=False)
ax[0].set_title(f"Training - {model_name}")
printA(cm_train)

print(f"--- {model_name} (Test) ---")
cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(cmap="Greys", ax=ax[1], colorbar=False)
ax[1].set_title(f"Test - {model_name}")
printA(cm_test)

# Show the figures
plt.tight_layout()
plt.show()

# %% Classifier 4
model_name = "Classifier 4"
model = xgb.XGBClassifier(max_depth=10,
                          n_estimators=8)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Scores
score_train = roc_auc_score(y_train, y_pred_train)
score_test = roc_auc_score(y_test, y_pred_test)

print(f"Training AUC: {score_train}")
print(f"Test AUC: {score_test}")

# Figures for the confusion matrices
fig, ax = plt.subplots(2, 1, figsize=(2, 4))
ax = ax.flatten()

# Confusion matrix for each output
print(f"--- {model_name} (Training) ---")
cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(cmap="Greys", ax=ax[0], colorbar=False)
ax[0].set_title(f"Training - {model_name}")
printA(cm_train)

print(f"--- {model_name} (Test) ---")
cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(cmap="Greys", ax=ax[1], colorbar=False)
ax[1].set_title(f"Test - {model_name}")
printA(cm_test)

# Show the figures
plt.tight_layout()
plt.show()

# %% Manual classifier
m = ExtraTreesClassifier()
m_name = "Tuned Classifier"
m, cm_train, cm_test = train_classifier(X_train,
                                        y_train,
                                        X_test,
                                        y_test,
                                        m_name,
                                        model=m)

# Confusion matrix for each output
print(f"--- {m_name} (Training) ---")
for cm in cm_train:
    printA(cm)

print(f"--- {m_name} (Test) ---")
for cm in cm_test:
    printA(cm)

# %% Classifier with hyperparameter tuning using Grid Search
# Check: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'max_depth': list(range(1, 10)),
    'min_child_weight': list(range(1, 10)),
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'scale_pos_weight': [1, 2, 3, 4, 5]
}

# Create the XGBoost model object
xgb_model = xgb.XGBClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(xgb_model,
                           param_grid,
                           cv=5,
                           scoring='roc_auc',
                           verbose=1,
                           n_jobs=-1
                           )

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# %% Try on test set
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)

print("Test Accuracy: %.5f%%" % (accuracy * 100.0))

# %% Multioutput classification
# Create the model
clf = xgb.XGBClassifier(eval_metric="auc",
                        objective="binary:logistic",
                        device="cuda:0")

# Hyperparameter tuning
param_grid = {
    'n_estimators': [i for i in range(10, 100, 5)],
    'max_depth': list(range(1, 10, 2)),
    'scale_pos_weight': [302 / 50000, 302 / 31885, 1, 2],
    'gamma': np.arange(0, 1., 0.2),
    'max_delta_step': np.arange(0, 5, 1),
}

clf = GridSearchCV(clf,
                   param_grid,
                   cv=5,
                   scoring='roc_auc',
                   n_jobs=-1,
                   verbose=3
                   )

# Train the model
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# Get the best model
best_clf = clf.best_estimator_

# Training accuracy
y_pred_train = best_clf.predict(X_train)
score = roc_auc_score(y_train, y_pred_train)
print(f"Training AUC: {score}")

# Test accuracy
y_pred_test = best_clf.predict(X_test)
score = roc_auc_score(y_test, y_pred_test)
print(f"Test AUC: {score}")

# Confusion matrices
cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
print(f"--- {m_name} (Training) ---")
printA(cm_train)

cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
print(f"--- {m_name} (Test) ---")
printA(cm_test)

# %% Save the model
# model_location = Path("models/TLCN/RESTART_model.json")
# best_clf.save_model(str(model_location))

# %% Load the model
model_name = "Tuned Classifier"
model_location = Path("models/TLCN/RESTART_model.json")
model = xgb.XGBClassifier()
model.load_model(str(model_location))

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Scores
score_train = roc_auc_score(y_train, y_pred_train)
score_test = roc_auc_score(y_test, y_pred_test)

# Figures for the confusion matrices
fig, ax = plt.subplots(2, 1, figsize=(2.5, 4), constrained_layout=True)
ax = ax.flatten()

# Confusion matrix for each output
print(f"--- {model_name} (Training) ---")
print(f"Training AUC: {score_train}")
# cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
# ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(cmap="Greys", ax=ax[0], colorbar=False)
# printA(cm_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, ax=ax[0], cmap="Greys", colorbar=False,
                                        normalize="pred")
ax[0].set_title(f"Training - {model_name}")

print(f"--- {model_name} (Test) ---")
print(f"Test AUC: {score_test}")
# cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
# ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(cmap="Greys", ax=ax[1], colorbar=False)
# printA(cm_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax[1], cmap="Greys", colorbar=False,
                                        normalize="pred")
ax[1].set_title(f"Test - {model_name}")

# Show the figures
fig.show()


# %% Test the loaded model
y_pred_test = loaded_model.predict(X_test)
score = roc_auc_score(y_test, y_pred_test)
print(f"Test AUC: {score}")

# %% # Get the rows where there's at least one zero in the outputs
y_with_zero = np.where(np.sum(y, axis=1) != y.shape[1], True, False)

# Get the indices of the rows with zero
indices_with_zero = np.where(y_with_zero)[0]

X_with_zero = X[indices_with_zero, :]

# %% Plot X
# ggplot style
plt.figure()

plt.plot(X_with_zero[:, :8], '-')
plt.xlim(0, 302)
plt.show()

# %% Uniques
X_with_zero_unique, indices_with_zero_unique = np.unique(X_with_zero, axis=0, return_index=True)

plt.figure()

plt.plot(X_with_zero_unique, '-')
plt.show()

# %%
import pickle
import os

plant_path = Path(
    os.getcwd()).resolve().parent / "CPS-SenarioGeneration/data/iptlc/RESTART/if1_dynamic_network_level_vulnerability/2024-05-09_15-09-08/plant.pkl"

with open(plant_path, "rb") as f:
    plant = pickle.load(f)

# %% dray a greyscale colorbar
fig, ax = plt.subplots(1, 1, figsize=(2, 4))
cax = ax.matshow(X_with_zero_unique, cmap='gray')
fig.colorbar(cax)
plt.show()
