#  Plant Disease Detection Using CNN

## Overview

This project uses a Convolutional Neural Network (CNN) to identify diseases in tomato plant leaves from images.

The model classifies images into four categories:

- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Healthy

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## Project Structure

```text
Plant-Disease-Detection/
│
├── src/
│   ├── config.py
│   ├── prepare_data.py
│   ├── train.py
│   └── predict.py
│
├── dataset/
├── model/
├── results/
├── test_images/
├── raw_data/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Workflow

```text
PlantVillage Dataset
        ↓
Data Preparation
        ↓
Train / Validation / Test
        ↓
CNN Model
        ↓
Model Training
        ↓
Evaluation
        ↓
Disease Prediction
```

## How to Run

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate environment

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare dataset

```bash
python src/prepare_data.py
```

### 5. Train model

```bash
python src/train.py
```

### 6. Predict an image

```bash
python src/predict.py test_images/leaf.jpg
```

### 7. Launch web application

```bash
streamlit run app.py
```

## Model

The CNN contains:

- Three convolutional layers
- Three max-pooling layers
- A flatten layer
- A dense layer
- A dropout layer
- A softmax output layer

Images are resized to 128 × 128 pixels and normalized before being passed to the model.

## Evaluation

The model generates:

- Training and validation accuracy graph
- Training and validation loss graph
- Confusion matrix
- Classification report

## Future Scope

The project can be extended by:

- Adding more plant species
- Adding additional diseases
- Using transfer learning
- Developing a mobile application
- Supporting real-time camera detection
- Providing treatment recommendations

## Conclusion

This project demonstrates the use of deep learning and CNNs for automated plant disease classification. It provides a simple foundation that can be expanded into a larger agricultural disease detection system.