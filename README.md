
# Image Classifier using SVM

This project implements an **image classification model** using **Support Vector Machine (SVM)** to classify images into two categories:

* **empty**
* **not\_empty**

## 📌 Steps Implemented

1. **Dataset Setup**

   * Images are stored in two folders inside the dataset directory (`empty` and `not_empty`).
   * Each image is read, resized to `15x15` pixels, and flattened into a feature vector.

2. **Feature & Label Creation**

   * The feature vectors (image data) are stored in `features`.
   * Labels are created as numerical values:

     * `0 → empty`
     * `1 → not_empty`.

3. **Train/Test Split**

   * The dataset is split into **training (80%)** and **testing (20%)** sets.
   * Stratified splitting is used to maintain class balance.

4. **Model Training with GridSearchCV**

   * An **SVM classifier (SVC)** is trained.
   * **Hyperparameter tuning** is done using `GridSearchCV` for parameters:

     * `C` (regularization) → `[1, 10, 100, 1000]`
     * `gamma` (kernel coefficient) → `[0.01, 0.001, 0.0001]`.
   * The best model is selected automatically.

5. **Model Evaluation**

   * The trained model is tested on the test set.
   * Accuracy is calculated and printed.

6. **Model Saving**

   * The best trained SVM model is saved as a `.pkl` file (`svm_model.pkl`) using `pickle`.
   * This model can be loaded later for predictions without retraining.

## ⚡ Requirements

* Python 3.x
* NumPy
* scikit-image
* scikit-learn


