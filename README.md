# Human Activity Recognition (HAR) and Gesture Analysis

This project demonstrates how to predict stock prices using Long Short-Term Memory (LSTM) models. The dataset used contains stock prices of NTT and includes fields such as open, close, high, low, volume, and daily percentage change. The model is designed to predict future stock prices based on past data, with additional features like moving averages and volatility to enhance performance.

## Project Overview

### Key Features:
- Self-Supervised Pretraining on HAR data with FFT-based data augmentation.
- LSTM Model captures sequential dependencies in time-series data.
- Transfer Learning adapts the model for gesture recognition.
- Automatic Data Handling for HAR and gesture datasets in .pt format.
 -Console Accuracy Output for evaluating model performance on gesture data.

## HAR Dataset:
  - Format: Provided as .pt files (PyTorch tensor format).
  - Content: Contains time-series data representing various human activities, typically captured through sensors such as accelerometers or gyroscopes.
  - Structure
    - Samples: Multidimensional array with dimensions representing samples, time steps, and features.
    - Labels: Integer labels indicating the activity type (e.g., walking, running, sitting).
  - Usage: Used in the pretraining phase with self-supervised learning and FFT-based augmentation to enhance model learning from unlabeled data.

## Gesture Recognition Dataset:
- Fields:
  - Format: .pt files similar to the HAR dataset.
  - Content: Contains time-series data of various hand gestures, potentially captured through wearable sensors or motion tracking systems.
  - Structure
    - Samples: Time-series samples with dimensions similar to the HAR dataset, where each sample represents a gesture sequence.
    - Labels: Class labels associated with each gesture type (e.g., wave, swipe, etc.).
  - Usage: Used in the fine-tuning phase to adapt the pretrained model specifically for gesture recognition tasks.

## Evaluation Metrics:
- Accuracy: Measures the percentage of correctly classified samples in the gesture dataset. It is used as the primary metric to evaluate the model’s performance after fine-tuning.
- Cross-Entropy Loss: Used during training to measure the difference between the predicted and actual class probabilities, guiding weight updates for classification accuracy.
- Contrastive Loss (Self-Supervised Learning): Applied in the pretraining phase on HAR data, focusing on distinguishing between augmented and non-augmented samples to improve the model's ability to recognize diverse features.

## Model Architecture

### LSTM Encoder:

- Input Layer: Takes time-series data with dimensions (samples, time steps, features).
- LSTM Layer: A single-layer LSTM (Long Short-Term Memory) unit with 128 hidden units, designed to capture temporal dependencies in the input sequences.
- Fully Connected Layers:
    - Dense Layer 1: A 128-unit layer with ReLU activation to refine features from the LSTM.
    - Output Layer: A softmax layer with a number of units equal to the number of activity classes (e.g., 6 for HAR data), producing class probabilities for classification.

### Self-Supervised Pretraining:

- Pretraining with unlabeled HAR data using FFT-based augmentation.
- The model is trained with contrastive loss to improve representation learning on activity patterns.

### Fine-Tuning for Gesture Recognition:

Once pretrained on HAR data, the model is fine-tuned with labeled gesture data for final classification, adapting the learned patterns for gesture-specific activities.

## Installation and Requirements

To run this project locally, you'll need to install the following dependencies:

```
pip install pytorch numpy
```

# Running the Project

To successfully run the project and replicate the results, follow these steps:

## Clone the Repository
First, clone this repository to your local machine using:

```
git clone https://github.com/soham2312/TimeSeries.git
cd TimeSeries
```

## Run the Python Script
Once the data is prepared, run the main script to process the data, build the LSTM model, and make predictions. Use this command to execute the script:

```
python main.py
```

## View the Output
Model Training:
The script will train an LSTM model on the historical stock data, with progress displayed during the epochs.

## Analyze Results
Once the model is trained and tested, the following metrics will be printed in the console:

```
Direction Accuracy: 78% (replace with actual value)
```

## Research Paper for Reference
https://wujns.edpsciences.org/articles/wujns/pdf/2022/06/wujns-1007-1202-2022-06-0521-10.pdf