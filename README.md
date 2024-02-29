## Cat vs Dog Image Classifier using Support Vector Machine (SVM)

This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs. The model is trained on a dataset containing images of cats and dogs and then evaluated on a separate test set.

## Dataset

The dataset used for training and testing the classifier consists of images of cats and dogs. It is divided into a training set and a test set. The images are resized to 64x64 pixels to facilitate processing.

The dataset directory structure is as follows:
1. dataset/
  training_set/
    cats/
      cat001.jpg
      cat002.jpg
      ...
    dogs/
      dog001.jpg
      dog002.jpg
      ...
test_set/
    cats/
      cat001.jpg
      cat002.jpg
      ...
    dogs/
      dog001.jpg
      dog002.jpg
      ...

Please replace "cat001.jpg", "dog001.jpg", etc., with your actual image filenames. Ensure that your directory structure matches the one mentioned in the README, where images of cats are placed under `training_set/cats/` and `test_set/cats/`, and images of dogs are placed under `training_set/dogs/` and `test_set/dogs/`. 

This structure ensures that the ImageDataGenerator can easily load the images for training and testing.



## Model Architecture

The SVM model is implemented using TensorFlow and scikit-learn. The input images are preprocessed and then fed into the SVM classifier. The model architecture includes:

- Convolutional layers to extract features from the images
- Pooling layers for dimensionality reduction
- Flatten layer to convert the 2D feature maps into a 1D feature vector
- SVM classifier for classification

## Training

The model is trained using the training set images. Image augmentation techniques such as rescaling, shearing, zooming, and horizontal flipping are applied to increase the diversity of the training data.

The training process involves optimizing the SVM classifier using the hinge loss function and the Adam optimizer. The training accuracy and loss are monitored to assess the model's performance.

## Evaluation

The trained model is evaluated using the test set images to assess its accuracy and generalization performance. Classification metrics such as accuracy, precision, recall, and F1-score are calculated to evaluate the model's performance.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed (TensorFlow, scikit-learn, etc.).
3. Run the provided Jupyter Notebook (`cat_dog_svm.ipynb`) to train and evaluate the model.
4. Use the trained model to classify new images of cats and dogs.

## Results

The model achieves a certain accuracy on the test set, demonstrating its effectiveness in classifying images of cats and dogs.





