# 🐕 Dog Breed Classifier

A sophisticated deep learning application that identifies dog breeds from images using PyTorch and EfficientNet. Features a beautiful Gradio web interface with confidence scoring and professional styling.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-5.43.1+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🎯 Accurate Breed Prediction**: Uses EfficientNet-B0 architecture for reliable dog breed classification
- **📊 Confidence Scoring**: Provides confidence levels with visual indicators
- **🎨 Beautiful UI**: Modern, responsive Gradio interface with custom CSS styling
- **📱 User-Friendly**: Simple drag-and-drop image upload with clear results
- **⚡ Fast Inference**: Optimized model for quick predictions
- **🔄 Interactive**: Clear button and real-time feedback

## 🚀 Live Demo

The application is designed to run locally and can be accessed through your web browser.

## 📁 Project Structure

```
DogBreedClassifier/
├── app.py              # Main Gradio web application
├── implementation.py    # PyTorch model implementation
├── code.ipynb          # Jupyter notebook with training code
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd DogBreedClassifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** (if you want to retrain)
   - The model was trained on the Oxford-IIIT Pet Dataset
   - Download from: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
   - Extract to a `dataset/` folder

4. **Download pre-trained model** (optional)
   - Place your trained model checkpoint in the project directory
   - Update the model path in `implementation.py` if needed

## 🎯 Usage

### Running the Web Application

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:7860`
   - The beautiful Dog Breed Predictor interface will load

3. **Upload an image**
   - Click the upload area or drag-and-drop a dog image
   - Click "🚀 Predict Breed" to get results

4. **View results**
   - See the predicted breed and confidence level
   - Get visual feedback based on confidence thresholds

### Using the Model Programmatically

```python
from implementation import predict_dog_breed
from PIL import Image

# Load an image
image = Image.open("dog_photo.jpg")

# Get prediction
breed, confidence = predict_dog_breed(image)
print(f"Predicted breed: {breed}")
print(f"Confidence: {confidence:.1f}%")
```

## 🧠 Model Architecture

- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Fine-tuning**: Custom classification head for dog breeds
- **Input**: RGB images (224x224 pixels)
- **Output**: Breed classification with confidence scores

## 📊 Training

The model training code is available in `code.ipynb`:

1. **Data Preparation**: Load and preprocess the Oxford-IIIT Pet Dataset
2. **Model Setup**: Initialize EfficientNet-B0 with custom classifier
3. **Training Loop**: Fine-tune the model on dog breed data
4. **Evaluation**: Assess model performance and save checkpoints

## 🔧 Configuration

### Model Parameters
- **Image Size**: 224x224 pixels
- **Batch Size**: 32 (adjustable)
- **Learning Rate**: 0.001 (adjustable)
- **Epochs**: 50 (adjustable)

### Web App Settings
- **Port**: 7860 (default Gradio port)
- **Host**: 0.0.0.0 (accessible from external connections)
- **Share**: False (local deployment)

## 📈 Performance

- **Accuracy**: High accuracy on dog breed classification
- **Speed**: Fast inference suitable for real-time applications
- **Memory**: Efficient model size for deployment

## 🎨 Customization

### Styling
The web interface uses custom CSS for a professional look:
- Modern color scheme with gradients
- Responsive design elements
- Custom confidence indicators
- Professional typography

### Model
- Easy to modify architecture in `implementation.py`
- Configurable confidence thresholds
- Extensible for additional breeds

## 🐛 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   - Ensure PyTorch is installed with CUDA support if using GPU
   - Fall back to CPU if GPU is not available

2. **Memory Issues**
   - Reduce batch size in training
   - Use smaller image dimensions if needed

3. **Port Already in Use**
   - Change the port in `app.py`
   - Kill existing processes on port 7860

### Error Messages
- **Low Confidence**: Upload a clearer, better-lit image
- **Model Loading Error**: Check model file path and format
- **Dependency Issues**: Verify all requirements are installed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Oxford-IIIT Pet Dataset** for training data
- **EfficientNet** architecture by Google Research
- **PyTorch** team for the deep learning framework
- **Gradio** for the web interface framework

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the code comments for implementation details

---

**Made with ❤️ using PyTorch and Gradio**

*Happy dog breed classifying! 🐕✨*
