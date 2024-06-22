# Fashion-MNIST-CNN-Dropout

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The CNN architecture includes dropout regularization to improve generalization and prevent overfitting. The model is trained using TensorFlow/Keras and evaluated based on several performance metrics.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Discussion](#discussion)
- [Contributions](#contributions)

## Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images of 28x28 pixels each, with 10 classes representing different types of clothing items. The dataset is divided into a training set of 60,000 images and a test set of 10,000 images.

### Classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Model Architecture
The CNN model includes the following layers:
1. **Convolutional Layers:** Extract features from the input images using filters with ReLU activation.
2. **Max-Pooling Layers:** Down-sample the input representation to reduce its dimensionality.
3. **Dropout Layers:** Prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
4. **Fully Connected (Dense) Layers:** Perform classification based on the extracted features.

### Model Summary:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```
## Usage
1. Ensure you have Jupyter Notebook or JupyterLab installed
2. Open the Jupyter Notebook
   ```bash
   jupyter notebook Fashion_mnist_CNN_with_dropout.ipynb
   ```
3. Run the notebook cells sequentially to preprocess data, build, train, and evaluate the model.

## Results
The trained CNN model achieved the following performance metrics on the test set:

-**Accuracy:** 91.39% <br>
-**Recall:** 91.39% <br>
-**Precision:** 91.53% <br>
-**F1-score:** 91.43% <br>
-**ROC AUC:** 99.48% <br>

## Discussion
The CNN model demonstrates high accuracy in classifying Fashion MNIST images. Dropout regularization effectively prevented overfitting, as evidenced by the model's performance on the validation set.

### Strengths:
The model's architecture captures the features of different clothing items well.
Dropout regularization ensures good generalization to new data.
### Challenges:
Some classes, such as T-shirts and shirts, may have subtle differences, leading to misclassifications.
Further tuning of hyperparameters and exploring advanced architectures could potentially improve performance. <br>

## Contributions
Contributions are welcome! Feel free to fork this repository and submit pull requests for any features, bug fixes, or improvements.
