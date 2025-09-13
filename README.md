# CIFAR-10 Image Classification with Deep Learning

## 🚀 Project Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The model is deployed as an interactive web application using Streamlit, allowing users to upload images and get real-time predictions.

## ✨ Features
- **Real-time Image Classification**: Upload any image and get instant predictions
- **10 CIFAR-10 Categories**: Classifies images into airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **User-friendly Interface**: Clean, intuitive Streamlit web interface
- **High Accuracy Model**: Custom CNN trained on CIFAR-10 dataset
- **Confidence Scoring**: Shows prediction confidence for each classification
- **No Setup Required**: Browser-based deployment for easy access

## 🔗 Live Demo
Try the live application: [CIFAR-10 Classifier](https://naveenkumar-2007-image-cnn-0tfygw.streamlit.app)

## 🛠️ Technical Implementation

### Model Architecture
- Custom Convolutional Neural Network built with TensorFlow/Keras
- Multiple Conv2D layers with MaxPooling for feature extraction
- Dense layers with Dropout for classification
- Optimized for CIFAR-10's 32x32 RGB images
- Trained on 50,000 training samples, validated on 10,000 test samples

### Web Application
- **Frontend**: Streamlit for interactive web interface
- **Backend**: TensorFlow for model inference
- **Image Processing**: Automatic resizing and normalization
- **Real-time Predictions**: Instant classification with confidence scores

## 📂 Project Structure
```
├── cnn.py              # Streamlit web application
├── cnn.h5              # Trained Keras model
├── names.pkl           # Class label mappings
├── Untitled21.ipynb    # Model training notebook
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🚀 Quick Start

### Local Setup
1. Clone the repository:
```bash
git clone https://github.com/Naveenkumar-2007/Classify-any-image-from-CIFAR-10-with-the-power-of-deep-learning-and-real-time-web-deployment.git
cd Classify-any-image-from-CIFAR-10-with-the-power-of-deep-learning-and-real-time-web-deployment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run cnn.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📋 Requirements
- Python 3.7+
- TensorFlow 2.x
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

All dependencies are listed in `requirements.txt`

## 🎯 Usage
1. Access the web application (locally or via the live demo)
2. Upload an image file (JPG, JPEG, or PNG)
3. Wait for the model to process your image
4. View the predicted class and confidence score
5. Try different images to test the model's performance

## 🧠 Model Training
The model was trained using the notebook `Untitled21.ipynb` with the following approach:
- **Dataset**: CIFAR-10 (60,000 32x32 color images)
- **Architecture**: Custom CNN with multiple convolutional and pooling layers
- **Training**: Supervised learning with categorical cross-entropy loss
- **Validation**: Achieved high accuracy on the test set
- **Export**: Model saved as `cnn.h5`, labels as `names.pkl`

## 🌐 Streamlit Deployment
The application is deployed on Streamlit Cloud for easy access:
- **Platform**: Streamlit Cloud
- **URL**: https://naveenkumar-2007-image-cnn-0tfygw.streamlit.app
- **Automatic Updates**: Synced with GitHub repository
- **Scalable**: Handles multiple concurrent users

## 🔮 Future Enhancements
- Support for additional image formats
- Batch processing capabilities
- Model performance metrics display
- Advanced preprocessing options
- Integration with other datasets

## 📊 CIFAR-10 Classes
The model can classify images into these 10 categories:
1. ✈️ Airplane
2. 🚗 Automobile
3. 🐦 Bird
4. 🐱 Cat
5. 🦌 Deer
6. 🐕 Dog
7. 🐸 Frog
8. 🐴 Horse
9. 🚢 Ship
10. 🚚 Truck

## 📄 License
This project is open source and available under the MIT License.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
⭐ Star this repo if you found it helpful!
