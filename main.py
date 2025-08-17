import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# dataset setup
dataset_path = r'D:\classifier\classification_data'
class_labels = ['empty', 'not_empty']

features = []
targets = []

# load images and create dataset
for label_idx, label in enumerate(class_labels):
    folder_path = os.path.join(dataset_path, label)
    for img_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, img_file)
        image = imread(file_path)
        image_resized = resize(image, (15, 15))
        features.append(image_resized.flatten())
        targets.append(label_idx)

features = np.array(features)
targets = np.array(targets)

# train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    features, targets, test_size=0.2, shuffle=True, stratify=targets
)

# initialize SVM classifier
svm_model = SVC()
param_grid = {
    'gamma': [0.01, 0.001, 0.0001],
    'C': [1, 10, 100, 1000]
}

grid_clf = GridSearchCV(svm_model, param_grid)
grid_clf.fit(X_train, Y_train)

# evaluate model
best_svm = grid_clf.best_estimator_
Y_pred = best_svm.predict(X_test)

accuracy = accuracy_score(Y_pred, Y_test)
print(f"{accuracy * 100:.2f}% accuracy on test set")

# save model
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(best_svm, model_file)

