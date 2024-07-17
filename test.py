import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from c import preprocess_image, extract_features, load_and_preprocess_image

# Load and preprocess the image
image_path = 'image.jpg'
image = load_and_preprocess_image(image_path)

# Extract features
features = extract_features(image)

X = np.array([features, features])  # Duplicate features to have at least two samples
y = np.array([0, 1])  # Dummy labels

# Manually split the data to ensure both classes are in training and test sets
X_train, X_test = X, X  # Since both are the same
y_train, y_test = y, y  # Since both are the same

svm_linear = SVC(kernel='linear')
svm_rbf = SVC(kernel='rbf')

# Train the SVM models
svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# Evaluate the SVM models
linear_score = svm_linear.score(X_test, y_test)
rbf_score = svm_rbf.score(X_test, y_test)

print(f'Test Accuracy (Linear): {linear_score}')
print(f'Test Accuracy (RBF): {rbf_score}')



