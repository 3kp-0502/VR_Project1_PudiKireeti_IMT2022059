import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# ===========================
# Step 1: Load and Process Data for ML Classifiers
# ===========================

data_dir = r"C:\Users\sanje\Downloads\Face-Mask-Detection-master\Face-Mask-Detection-master\dataset"
categories = ["with_mask", "without_mask"]

# HOG and LBP Parameters
hog_params = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'}
lbp_radius = 1
lbp_n_points = 8 * lbp_radius

def load_data():
    X, y = [], []
    sample_images = []
    sift = cv2.SIFT_create()

    for label, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        for file in tqdm(os.listdir(folder_path), desc=f"Processing {category}"):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128))

            # HOG feature extraction
            hog_features, hog_image = hog(img, visualize=True, **hog_params)

            # LBP feature extraction
            lbp = local_binary_pattern(img, lbp_n_points, lbp_radius, method="uniform")
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_n_points + 3), density=True)

            # SIFT feature extraction
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is None:
                descriptors = np.zeros((1, 128))
            sift_features = np.mean(descriptors, axis=0)

            # Combine features
            features = np.hstack((hog_features, lbp_hist, sift_features))
            X.append(features)
            y.append(label)

            # Store sample images for visualization
            if len(sample_images) < 2:
                sift_img = cv2.drawKeypoints(img, keypoints, None)
                sample_images.append((img, hog_image, lbp, sift_img, category))

    # Apply PCA for dimensionality reduction
    X = np.array(X)
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)

    # Show sample images and extracted features
    fig, axes = plt.subplots(len(sample_images), 4, figsize=(16, 8))
    for idx, (orig, hog_feat, lbp_feat, sift_feat, label) in enumerate(sample_images):
        axes[idx, 0].imshow(orig, cmap='gray')
        axes[idx, 0].set_title(f"Original - {label}")
        axes[idx, 1].imshow(hog_feat, cmap='gray')
        axes[idx, 1].set_title("HOG Features")
        axes[idx, 2].imshow(lbp_feat, cmap='gray')
        axes[idx, 2].set_title("LBP Features")
        axes[idx, 3].imshow(sift_feat, cmap='gray')
        axes[idx, 3].set_title("SIFT Keypoints")
    plt.show()

    return X_pca, np.array(y)

# Load dataset for ML Classifiers
X, y = load_data()

# Split for ML classifiers
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ===========================
# Step 2: Train and Evaluate ML Classifiers
# ===========================

classifiers = {
    "SVM": SVC(kernel='linear'),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
hyperparams = {}

for name, model in classifiers.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    results[name] = accuracy
    hyperparams[name] = str(model.get_params())
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(f"Hyperparameters:\n{model.get_params()}")
    print(classification_report(y_test, preds))

# ===========================
# Step 3: Load Raw Data for CNN (Direct Images)
# ===========================

def load_raw_images():
    images, labels = [], []
    for label, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

X_cnn, y_cnn = load_raw_images()
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn.reshape(-1, 128, 128, 1), y_cnn, test_size=0.2, random_state=42)

# ===========================
# Step 4: CNN Training
# ===========================

def build_cnn(learning_rate=0.001, activation='relu', optimizer='adam'):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation=activation, input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation=activation),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation=activation),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_hyperparams = [
    {'learning_rate': 0.001, 'activation': 'relu'},
    {'learning_rate': 0.01, 'activation': 'relu'},
    {'learning_rate': 0.001, 'activation': 'sigmoid'}
]

for params in cnn_hyperparams:
    cnn_model = build_cnn(**params)
    history = cnn_model.fit(X_train_cnn, y_train_cnn, epochs=5, batch_size=32, validation_split=0.2)
    acc = cnn_model.evaluate(X_test_cnn, y_test_cnn)[1]
    results[f"CNN {params}"] = acc
    hyperparams[f"CNN {params}"] = str(params)
    print(f"\nCNN Accuracy: {acc:.4f} | Params: {params}")

# ===========================
# Step 5: Compare Results
# ===========================

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(results.keys(), results.values(), color='skyblue')

# Add hyperparameter labels to each bar
for bar, (name, acc) in zip(bars, results.items()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05,
            f"{acc:.2%}\n{hyperparams[name][:40]}", ha='center', fontsize=8)

plt.xticks(rotation=45, ha='right')
plt.title("Classifier and CNN Performance Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()
