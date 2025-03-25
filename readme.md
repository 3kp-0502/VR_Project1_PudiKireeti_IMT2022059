**Face Mask Detection, Classification, and Segmentation**

***

***Introduction:***

This project aims to develop a computer vision solution for detecting
and segmenting face masks in images. The approach involves both
traditional machine learning methods and deep learning techniques to
achieve accurate classification and segmentation.

-   **Binary Classification Using Handcrafted Features and ML
    Classifiers**: Traditional image processing techniques are used to
    extract handcrafted features, followed by machine learning
    classifiers such as SVM, Random Forest for mask classification.

-   **Binary Classification Using CNN**: A Convolutional Neural Network
    (CNN) is trained to classify images into two categories---mask and
    no-mask---leveraging deep learning for feature extraction and
    classification.

-   **Region Segmentation Using Traditional Techniques**: Classical
    image processing methods such as thresholding, edge detection are
    explored for segmenting mask regions.

-   **Mask Segmentation Using U-Net**: A U-Net architecture is
    implemented for pixel-wise mask segmentation, ensuring precise
    localization of masks within images.

This combination of techniques allows for a comprehensive evaluation of
different approaches in mask classification and segmentation, providing
insights into the strengths of traditional vs. deep learning methods.

***

***Dataset:***

This project utilizes two datasets for face mask classification and
segmentation:

1.  **Face Mask Detection Dataset**

    -   **Source**: [GitHub - Face Mask
        Detection](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)

    -   **Description**: This dataset consists of labelled images of
        individuals with and without face masks. It is used for binary
        classification tasks to distinguish between masked and unmasked
        faces.

    -   **Structure**:

        -   *With Mask*: Contains images of people wearing face masks.

        -   *Without Mask*: Contains images of people without face
            masks.

2.  **Masked Face Segmentation Dataset (MFSD)**

    -   **Source**: [GitHub - MFSD](https://github.com/sadjadrz/MFSD)

    -   **Description**: This dataset includes images of faces with
        corresponding ground truth mask segmentation labels. It is used
        for segmentation tasks to accurately delineate mask regions in
        images.

These datasets serve as the foundation for training and evaluating both
traditional machine learning and deep learning models for face mask
classification and segmentation.

***

***Methodology:***

***Classification:***

Here's a detailed methodology and comparison of the machine learning
classifiers and CNN-based model for binary face mask classification.

**1. Feature Extraction & Data Preprocessing**

**1.1 Dataset Description**

The dataset consists of two categories:

-   **with_mask** (labeled as 0)

-   **without_mask** (labeled as 1)

Images are preprocessed into grayscale and resized to **128x128
pixels**.

**1.2 Feature Extraction Methods**

We extracted handcrafted features using:

-   **Histogram of Oriented Gradients (HOG)**

-   **Local Binary Patterns (LBP)**

-   **Scale-Invariant Feature Transform (SIFT)**

Each image was represented as a feature vector, combining these three
descriptors.

**HOG Features**

-   Captures **gradient-based structural features** by computing
    gradients in small image regions.

-   **Parameters:**

    -   orientations=9

    -   pixels_per_cell=(8,8)

    -   cells_per_block=(2,2)

**LBP Features**

-   Captures **texture patterns** by comparing pixel intensity with
    surrounding neighbors.

-   **Parameters:**

    -   radius=1

    -   n_points=8 \* radius

    -   method=\"uniform\"

**SIFT Features**

-   Extracts **keypoints** and represents images in a scale- and
    rotation-invariant manner.

-   Since the number of keypoints varies across images, the feature
    vector is **averaged**.

After feature extraction, **Principal Component Analysis (PCA)** was
applied to reduce dimensionality to **100 components**.

***

**2. Machine Learning Classifiers**

The extracted features were used to train the following classifiers:

**2.1 Support Vector Machine (SVM)**

-   Uses a linear kernel to **find the best decision boundary**.

-   Trained on extracted features.

-   Achieved **75.58% accuracy**.

**2.2 Multilayer Perceptron (MLP)**

-   A **fully connected feedforward neural network** with a single
    hidden layer of **100 neurons**.

-   Trained using **backpropagation**.

-   Achieved **83.19% accuracy**, the best among ML models.

**2.3 Random Forest**

-   An **ensemble of decision trees**, using **100 estimators**.

-   Captures **non-linear patterns** well.

-   Achieved **79.88% accuracy**.

**2.4 Decision Tree**

-   A simple **tree-based model**.

-   Tends to overfit on complex data.

-   Achieved **65.64% accuracy**, the lowest performance.

**2.5 k-Nearest Neighbors (KNN)**

-   Classifies samples based on the **majority vote** of their **5
    nearest neighbors**.

-   Achieved **73.37% accuracy**.

***

**3. CNN-Based Classification**

To improve performance, we trained a **Convolutional Neural Network
(CNN)** directly on image data.

**3.1 CNN Architecture**

-   **Conv2D (32 filters, 3×3, activation=ReLU)**

-   **MaxPooling2D (2×2)**

-   **Conv2D (64 filters, 3×3, activation=ReLU)**

-   **MaxPooling2D (2×2)**

-   **Conv2D (128 filters, 3×3, activation=ReLU)**

-   **MaxPooling2D (2×2)**

-   **Flatten**

-   **Dense (128 neurons, activation=ReLU)**

-   **Dense (1 neuron, activation=sigmoid)**

**3.2 CNN Hyperparameter Tuning**

We trained CNN with different settings:

1.  **Learning rate = 0.001, Activation = ReLU** → **Accuracy: 90.92%**

2.  **Learning rate = 0.01, Activation = ReLU** → **Accuracy: 55.58%**

3.  **Learning rate = 0.001, Activation = Sigmoid** → **Accuracy:
    72.34%**

CNN with a **0.001 learning rate and ReLU activation** performed the
best.

***

## 4. Performance Comparison

| Model                                          | Accuracy  |
|-----------------------------------------------|-----------|
| **CNN (LR=0.001, ReLU)**                      | **90.92%** |
| MLP                                           | 83.19%    |
| Random Forest                                 | 79.88%    |
| SVM                                           | 75.58%    |
| KNN                                           | 73.37%    |
| Decision Tree                                 | 65.64%    |

![WhatsApp Image 2025-03-25 at 19 32 34_1a777c10](https://github.com/user-attachments/assets/b9ce1c3c-cf80-4da6-b407-494087d6a527)

**Key Observations**

1.  **CNN significantly outperforms ML classifiers** due to automatic
    feature learning.

2.  **MLP performs the best among ML models**, likely due to its **deep
    network structure**.

3.  **Random Forest and SVM perform decently**, while **KNN and Decision
    Tree lag behind**.

4.  **Hyperparameter tuning in CNN is crucial**---a **high learning rate
    (0.01) led to poor results (55.58%)**.

5.  **CNN (90.92%) is the best approach** for face mask classification.

6.  **MLP (83.19%) is the strongest among traditional ML classifiers**.

7.  **Feature extraction methods (HOG, LBP, SIFT) were effective**, but
    CNN **eliminates the need for manual feature engineering**.

8.  **Hyperparameter tuning (learning rate, activation function) plays a
    major role in CNN performance**.

***

***Segmentation:***

**Methodology**

This section outlines the steps taken to implement region-based
segmentation using traditional techniques and U-Net for mask
segmentation. The methodology includes data preprocessing, feature
extraction, model training, and evaluation of segmentation techniques.

**1. Data Preprocessing and Feature Extraction**

To ensure uniformity in image size and quality, all images were
preprocessed before segmentation. The preprocessing steps included:

-   **Reading Images:** Images were loaded in grayscale using OpenCV
    (cv2.imread) to simplify processing.

-   **Resizing:** Each image was resized to **128 × 128** pixels to
    ensure consistency in model training and segmentation.

-   **Normalization:** Pixel values were normalized to the range \[0,1\]
    by dividing by 255, improving model convergence and thresholding
    performance.

-   **RGB Conversion:** When required, grayscale images were stacked
    into three channels to match the input shape expected by U-Net.

***

**2. Region-Based Segmentation (Thresholding and Edge Detection)**

Traditional segmentation techniques were applied to extract mask regions
from face images:

-   **Thresholding (Otsu's Method):**

    -   Applied Otsu's thresholding (cv2.THRESH_BINARY +
        cv2.THRESH_OTSU) to segment the mask regions.

    -   Automatically determined an optimal threshold value to separate
        foreground and background.

-   **Edge Detection (Canny Edge Detection):**

    -   Applied cv2.Canny() to extract edge boundaries of the mask
        regions.

    -   Used threshold values of 100 and 200 for detecting strong and
        weak edges, respectively.

***

**3. U-Net Model for Mask Segmentation**

To improve segmentation accuracy, a U-Net model was implemented with the
following architecture:

-   **Encoder Path:**

    -   Successive **Conv2D** layers with ReLU activation and
        **MaxPooling2D** layers to capture spatial features.

-   **Decoder Path:**

    -   **UpSampling2D** layers followed by **Conv2D** layers to
        reconstruct the segmented mask.

-   **Skip Connections:**

    -   Used concatenation (layers.concatenate) to merge low-level
        features from the encoder to retain fine-grained details.

-   **Final Output Layer:**

    -   A **1 × 1 Conv2D** layer with a **sigmoid** activation function
        was used to output a binary segmentation mask.

***

**4. Model Training and Evaluation**

-   **Training:**

    -   The dataset was split into **training (80%) and testing (20%)**
        subsets.

    -   The U-Net model was compiled with the **Adam optimizer** and
        trained using the **binary cross-entropy loss function**.

    -   Trained for **3 epochs** with a batch size of **8**.

-   **Evaluation Metrics:**

    -   **Intersection over Union (IoU):** Measures overlap between the
        predicted and ground truth masks.

    -   **Dice Coefficient:** Measures similarity between segmentation
        results and ground truth.

    -   **Accuracy:** Percentage of correctly classified pixels in the
        segmentation output.

***

**5. Visualization and Performance Comparison**

The segmentation results were visualized and compared across different
techniques:

-   **Original Image**

-   **Ground Truth Mask**

-   **Thresholding Result**

-   **Edge Detection Result**

-   **U-Net Segmentation Result**

![WhatsApp Image 2025-03-24 at 22 56 17_a8d7ac46](https://github.com/user-attachments/assets/f3900fea-0a9b-4d9a-a11e-d73c09819d64)
![WhatsApp Image 2025-03-24 at 22 58 03_2c4eff83](https://github.com/user-attachments/assets/52f51e45-fca7-4a3f-bbc5-1a582ce8abf4)
![WhatsApp Image 2025-03-24 at 22 58 20_cb069c98](https://github.com/user-attachments/assets/af7228aa-53a5-40e7-a606-10b88c47ce85)


Performance metrics were computed for each method, demonstrating that
**U-Net outperformed traditional techniques** in mask segmentation. The
final comparison revealed that **U-Net achieved a higher IoU and Dice
score** than thresholding and edge detection, validating its
effectiveness for precise mask segmentation.

***

***Hyperparameters and Experiments:***

Here\'s a structured explanation of the **Hyperparameters and
Experiments** for the CNN model, including the variations tried and
their impact on performance.

**Hyperparameters for CNN**

In this study, we designed a **Convolutional Neural Network (CNN)** to
classify images as \"with mask\" or \"without mask.\" The following
hyperparameters were considered:

**1. Learning Rate (LR)**

-   **Definition**: Controls the step size at which the model updates
    weights during training.

-   **Values Tried**:

    -   **0.001** (default, commonly used)

    -   **0.01** (higher, aggressive updates)

**2. Activation Function**

-   **Definition**: Defines the non-linearity introduced in the model.

-   **Values Tried**:

    -   **ReLU (Rectified Linear Unit)** (default, avoids vanishing
        gradients)

    -   **Sigmoid** (commonly used for binary classification)

**3. Optimizer**

-   **Definition**: Defines how the model updates weights during
    training.

-   **Values Tried**:

    -   **Adam** (default, adaptive learning)

**4. Batch Size**

-   **Definition**: The number of training samples per iteration.

-   **Values Tried**:

    -   **32** (standard, balanced training)

**5. Number of Epochs**

-   **Definition**: Number of times the model goes through the entire
    dataset.

-   **Fixed at 5** (due to time constraints, but typically a higher
    value is better)

**Experiments and Results**

  ## Experiment Results

| Experiment | Learning Rate | Activation | Optimizer | Accuracy (%) |
|-----------|--------------|------------|-----------|--------------|
| **CNN-1** | 0.001        | ReLU       | Adam      | **90.92**    |
| **CNN-2** | 0.01         | ReLU       | Adam      | **55.58**    |
| **CNN-3** | 0.001        | Sigmoid    | Adam      | **55.58**    |


**Observations**

1.  **ReLU performed better** than Sigmoid, confirming that it avoids
    vanishing gradients.

2.  **Higher learning rate (0.01) led to instability**, causing accuracy
    to drop.

3.  The best configuration was **CNN-1 (LR = 0.001, ReLU, Adam),
    achieving 90.92% accuracy**.

4.  Hyperparameter tuning showed that **ReLU with Adam and a learning
    rate of 0.001 is optimal**.

***

**Hyperparameters and Experiments for U-Net Model**

**1. Base Model Hyperparameters**

-   **Input Size: 128 × 128 × 3**

-   **Number of Filters in Encoder: 64, 128, 256**

-   **Kernel Size: 3 × 3**

-   **Activation Function: ReLU**

-   **Pooling Layers: MaxPooling with a 2 × 2 window**

-   **Upsampling Layers: 2 × 2 factor**

-   **Final Activation: Sigmoid**

-   **Optimizer: Adam**

-   **Loss Function: Binary Cross-Entropy**

-   **Batch Size: 8**

-   **Epochs: 3**

**Additional Implicit Parameters**

-   **Learning Rate: Default (0.001 for Adam)**

-   **Pooling Stride: Default stride of (2, 2) for MaxPooling2D**

-   **Upsampling Interpolation: Nearest-neighbor (default for
    UpSampling2D)**

-   **Padding: \"same\" for all convolutional layers**

**2. Experimentation and Results**

  ## Experiment Results

| Experiment      | Learning Rate | Batch Size | Epochs | IoU       | Dice Score | Accuracy  |
|---------------|--------------|-----------|--------|-----------|------------|-----------|
| **Base Model**  | 0.001        | 8         | 3      | **0.7021** | **0.8913** | **0.5929** |
| **Experiment 1**| 0.0001       | 8         | 5      | **0.6784** | **0.8725** | **0.5891** |
| **Experiment 2**| 0.001        | 16        | 3      | **0.7053** | **0.8947** | **0.6012** |


**Observations:**

-   Lowering the learning rate to 0.0001 slowed training without
    significant improvement.

-   Increasing epochs to 5 caused slight overfitting, leading to a
    marginal performance drop.

-   Increasing batch size to 16 resulted in a small accuracy gain but no
    major IoU/Dice improvement.

***Results:***

***Classification:***

**1. Introduction**

This section presents the evaluation metrics for all tested approaches,
including Convolutional Neural Networks (CNNs) and traditional machine
learning models. The models were evaluated based on **accuracy** to
compare their performance on the given task.

**2. CNN Experiments**

Three variations of CNN models were tested by changing the **learning
rate** and **activation function** while keeping the optimizer constant
(**Adam**). The results are summarized in the table below:

  ## Experiment Results

| Experiment | Learning Rate | Activation Function | Optimizer | Accuracy (%) |
|-----------|--------------|--------------------|-----------|--------------|
| **CNN-1** | 0.001        | ReLU               | Adam      | **90.92**    |
| **CNN-2** | 0.01         | ReLU               | Adam      | **55.58**    |
| **CNN-3** | 0.001        | Sigmoid            | Adam      | **55.58**    |


**Observations**

-   The best-performing CNN model (**CNN-1**) achieved **90.92%
    accuracy**, using **ReLU activation** and a **0.001 learning rate**.

-   Increasing the learning rate to **0.01 (CNN-2)** led to a
    significant drop in accuracy (**55.58%**), likely due to unstable
    weight updates.

-   Using **Sigmoid activation (CNN-3)** also resulted in **55.58%
    accuracy**, indicating ReLU is a better activation function for this
    task.

**3. Machine Learning Model Comparisons**

We also evaluated traditional machine learning models for comparison.
The results are as follows:

  ## Model Performance Comparison

| Model            | Accuracy (%) |
|-----------------|--------------|
| **SVM**         | **75.58%**    |
| KNN            | 73.37%        |
| Decision Tree  | 65.64%        |
| **MLP** (Neural Network) | **83.19%**    |
| Random Forest  | 79.88%        |


**Observations**

-   Among the ML models, **MLP (83.19%) and SVM (75.58%)** achieved the
    highest accuracy.

-   **Random Forest (79.88%)** and **KNN (73.37%)** also performed
    reasonably well.

-   **Decision Tree (65.64%)** had the lowest accuracy among traditional
    models, likely due to overfitting on training data.

**4. Conclusion**

-   **CNN outperforms traditional ML models**, achieving the highest
    accuracy (**90.92%**) when using **ReLU activation and a 0.001
    learning rate**.

-   **MLP (83.19%) and SVM (75.58%)** are the best alternatives among
    machine learning approaches.

-   **Hyperparameter tuning significantly affects CNN performance**---a
    high learning rate led to poor accuracy.

-   **ReLU activation is better than Sigmoid** in this case, as Sigmoid
    led to reduced performance.

***Segmentation:***

**Evaluation Metrics Comparison**

The following table presents the evaluation metrics for different
approaches, including U-Net, thresholding, and edge detection.

  ## Approach Performance Comparison

| Approach        | Mean IoU  | Mean Dice | Mean Accuracy |
|---------------|----------|-----------|--------------|
| **U-Net**     | **0.7021** | **0.8913**  | **0.5929**  |
| Thresholding  | 0.3002   | 0.4593    | 0.3079      |
| Edge Detection | 0.1205   | 0.2343    | 0.5479      |

![WhatsApp Image 2025-03-24 at 22 56 30_6811343f](https://github.com/user-attachments/assets/3ad1f851-1528-44a4-977f-6f2e22c164b1)


**Observations:**

-   **U-Net** achieved the highest performance across all metrics, with
    a **Mean IoU of 0.7021**, **Mean Dice of 0.8913**, and **Mean
    Accuracy of 0.5929**.

-   **Thresholding** performed moderately but showed significantly lower
    IoU and Dice scores compared to U-Net.

-   **Edge Detection** had the lowest IoU and Dice scores, but its
    accuracy (0.5479) was closer to U-Net than thresholding, likely due
    to correct edge identification in some cases.

***Observations and Analysis:***

The results indicate that deep learning approaches, particularly CNNs
and U-Net architectures, significantly outperform traditional machine
learning models in both classification and segmentation tasks.
Hyperparameter tuning plays a critical role in optimizing model
performance, as evidenced by the starp differences observed with varying
learning rates and activation functions.

In summary:

-   CNNs provide superior performance over traditional ML models.

-   Hyperparameter choices directly affect model accuracy.

-   U-Net excels in segmentation tasks compared to simpler methods like
    thresholding and edge detection.

These insights underline the importance of selecting appropriate models
and tuning hyperparameters carefully to achieve optimal results in
machine learning applications.

**Challenges Faced & How They Were Addressed**

1.  **Dataset Download Took More Time**

    -   **Challenge:** The dataset was large, leading to slow downloads
        and delays in preprocessing.

    -   **Solution:** We used a faster internet connection

2.  **Long Training Times**

-   **Challenge:** Training deep learning models, especially U-Net and
    CNNs, was time-consuming due to high computational requirements.

-   **Solution:** We optimized training by using **GPU acceleration
    (CUDA)**

3.  **Hyperparameter Tuning Complexity**

-   **Challenge:** Choosing the optimal learning rate, activation
    function, and optimizer required multiple experiments, increasing
    computational costs.

-   **Solution:** We used **GridSearchCV for ML models**

4.  **Difficulty in Edge Detection-Based Segmentation**

-   **Challenge:** Edge detection struggled with complex boundaries,
    leading to poor segmentation results.

-   **Solution:** We experimented with **Canny edge detection and Sobel
    filtering but found U-Net to be the best alternative**.

***How to Run the Code:***

**1. Prerequisites**

Before running the scripts, ensure you have the following installed on
your system:

-   Python (version 3.6 or higher)

-   Required libraries:

    -   OpenCV

    -   NumPy

    -   Matplotlib

    -   Scikit-image

    -   Scikit-learn

    -   TensorFlow (for CNN)

You can install the required libraries using pip if you haven\'t done
so:

pip install opencv-python numpy matplotlib scikit-image scikit-learn
tensorflow

**2. Prepare Dataset for Classification**

-   Download or create a dataset for face mask detection.

-   Update the data_dir variable in the classification code to point to
    your dataset path:

data_dir =
r\"C:\\Users\\sanje\\Downloads\\Face-Mask-Detection-master\\Face-Mask-Detection-master\\dataset\"

**3. Run the Classification Code**

-   Open your preferred Python IDE or text editor (e.g., Jupyter
    Notebook, PyCharm, VSCode).

-   Copy the classification code into a new Python file
    (e.g., classification.py).

-   Execute the script by running:

python classification.py

-   The script will load the dataset, extract features using HOG, LBP,
    and SIFT methods, train various classifiers (SVM, MLP, Random
    Forest, Decision Tree, KNN), and display accuracy results.

**4. Prepare Image for Segmentation**

-   Ensure you have an image ready for segmentation
    (e.g., zoomed_image.png).

-   Update the image_path variable in the segmentation code to point to
    your image path:

image_path =
r\"C:\\Users\\sanje\\Downloads\\stitched_image\\zoomed_image.png\"

-   Update the output_folder variable to specify where you want to save
    processed images:

output_folder = r\"C:\\Users\\sanje\\Downloads\\zoomed\"

**5. Run the Segmentation Code**

-   Open another Python file (e.g., segmentation.py) and copy the
    segmentation code into it.

-   Execute the script by running:

python segmentation.py

-   The script will process the specified image, apply Gaussian blur and
    Otsu\'s thresholding, perform morphological operations, detect
    contours, and save the processed image with axon details.

**6. Review Results**

-   After running both scripts:

    -   For classification, check console outputs for classifier
        accuracies and classification reports.

    -   For segmentation, review printed axon details and check the
        output folder for processed images.
