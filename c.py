import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

def preprocess_image(image, target_size=(150, 110)):
    return cv2.resize(image, target_size)

def compute_gradients(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    return magnitude, angle

def g_ltp(image, threshold=10):
    magnitude, _ = compute_gradients(image)
    g_ltp = np.zeros_like(image, dtype=np.int8)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            center_val = magnitude[i, j]
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if x == 0 and y == 0:
                        continue
                    neighbor_val = magnitude[i+x, j+y]
                    if neighbor_val > center_val + threshold:
                        g_ltp[i, j] = 1
                    elif neighbor_val < center_val - threshold:
                        g_ltp[i, j] = -1
    return g_ltp

def dglp(image, num_directions=16):
    _, angle = compute_gradients(image)
    dglp_hist = np.zeros((num_directions,), dtype=np.int32)
    direction_size = 360 // num_directions
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            direction = int(angle[i, j] // direction_size) % num_directions
            dglp_hist[direction] += 1
    return dglp_hist

def extract_features(image):
    g_ltp_features = g_ltp(image)
    dglp_features = dglp(image)
    combined_features = np.concatenate([g_ltp_features.flatten(), dglp_features])
    return combined_features

def load_and_preprocess_image(image_path, target_size=(150, 110)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    preprocessed_image = preprocess_image(image, target_size)
    return preprocessed_image

