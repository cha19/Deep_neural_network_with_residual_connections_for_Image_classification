<!-- # Group14_CS584(Machine learning)
This repository is about the project which is part of CS584 (Machine Learning). The objective of this project is to construct a custom model with lesser parameters and higher performance with skip connections and depth scaling principals. We have considered 2 SOTA models(ResNet50 and VGG19) and one custom model in this project. 

Dataset 1: Butterfly and moth species (https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)  
Dataset 2: FER (https://www.kaggle.com/datasets/msambare/fer2013)  
ML14a: Model training on Butterfly dataset  
ML14b: Model training on FER dataset

# Problem 
The realm of very deep neural networks is marked by its immense potential for intricate tasks in computer vision. However, two significant challenges cast shadows on their widespread adoption.

# a. Vanishing Gradient:
As the depth of a neural network increases, the backpropagation of errors during training encounters a critical hurdle known as the vanishing gradient problem [10, 11, 12]. This challenge arises due to the diminishing magnitude of gradients as they traverse through numerous layers during backpropagation. In essence, the gradients dwindle to near-zero values, rendering them ineffective in updating the weights of early layers. Consequently, these layers fail to learn meaningful representations, impeding the overall convergence of the network. The vanishing gradient problem becomes particularly pronounced in very deep architectures, where the prolonged path that gradients traverse exacerbates the attenuation effect.

# b. Long Computational Time:
Simultaneously, the computational demands associated with training very deep neural networks escalate dramatically. The intricate interplay of an increasing number of layers and parameters necessitates extensive computational resources and time. The sheer complexity of the network structure contributes to prolonged training times, making it a formidable challenge to efficiently harness the potential of these deep architectures within reasonable timeframes. The extended training periods not only strain computational resources but also hinder the swift development and deployment of models for practical applications.

To address these challenges, we have introduced a custom model with fewer parameters and enhanced performance. The details of the proposed algorithm are discussed in Section 3 of our project report.  Subsequently, Model training and experimentation, Results, and discussions are presented in Sections 4 and 5, respectively.  -->





# Facial Emotion Recognition using VGG16 and Transfer Learning

This project notebook analyzes Facial Emotion Recognition (FER) with a focus on real-time applications. Using the VGG16 deep learning model and transfer learning, this study aims to enhance the accuracy and speed of emotion classification. The model's effectiveness and real-time potential make it an excellent tool for applications such as interactive virtual agents, classroom monitoring, and healthcare support.

Facial Emotion Recognition

## Project Summary

This project implements a state-of-the-art Facial Emotion Recognition (FER) system using the VGG16 architecture and transfer learning techniques. Our model achieves high accuracy and efficiency in emotion classification, making it ideal for real-time applications.

### Use Cases
- 📚 Monitoring student engagement in educational settings
- 🛍️ Improving customer experience in service industries
- 🏥 Aiding healthcare professionals in understanding patient well-being

### Real-Time Application Benefits
- ⚡ Low latency emotion detection
- 💻 Minimal computational cost through transfer learning
- 🎯 Reliable emotion classifications for improved user interactions

## Approach

1. **Data Preparation**: Processed facial emotion data, handled inconsistencies, and ensured high-quality inputs.
2. **Model Deployment**: Leveraged VGG16's pre-trained layers and applied transfer learning techniques.
3. **Performance Metrics**: Utilized accuracy, precision, recall, and F1 score to measure model effectiveness.


## Usage

```python
from fer_model import FERModel

# Initialize the model
model = FERModel()

# Predict emotion from an image
emotion = model.predict('path/to/image.jpg')
print(f"Detected emotion: {emotion}")
```

## Code Overview

### 1. Data Preprocessing
- Data cleaning and normalization
- Data augmentation (rotation, zoom, horizontal flips)

### 2. Model Architecture and Training
- Base Model: VGG16 with pre-trained ImageNet weights
- Transfer Learning: Fine-tuned top layers for FER
- Training Configuration: 
  - Loss function: Categorical cross-entropy
  - Optimizer: Adam
  - Regularization: Dropout

### 3. Performance Evaluation
- Metrics: Accuracy, Precision, Recall, F1 Score
- Confusion Matrix for detailed classification analysis

## Results and Insights

Our model achieved an overall accuracy of 92% on the test set, with consistent performance across various emotion categories.

Confusion Matrix

## Dataset

We used the FER2013 dataset, which contains 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.

Dataset structure:
- 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Training set: 28,709 examples
- Public test set: 3,589 examples


```
## 📧 Let’s Connect!
If you're excited about Machine Learning, AI, or data-driven innovation, I’d love to connect! Whether it’s brainstorming ideas, collaborating on projects, or just geeking out over cool models, feel free to reach out.

📬 Email: saicharan.gangili@gmail.com

Let’s build something amazing together! 🚀🤖
