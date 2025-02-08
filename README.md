# Sentiment Analysis using CNN-LSTM

## Overview
This project implements a **Sentiment Analysis model using CNN-LSTM**. The model leverages **Convolutional Neural Networks (CNNs)** to extract local features and **Long Short-Term Memory (LSTM)** networks to capture long-term dependencies in text data. The goal is to classify text into positive, negative, or neutral sentiment categories.

## Installation & Setup
To set up and run the project, follow these steps:

1. Clone this repository:
   ```sh
   git clone https://github.com/ghayazzz/cnn-lstm.git
   cd cnn-lstm
   ```
2. Install required dependencies manually:
   ```sh
   pip install tensorflow keras nltk scikit-learn matplotlib numpy pandas
   ```
   (Ensure you have Python 3.7+ installed.)

3. Download necessary NLTK resources:
   ```sh
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

4. Launch Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
5. Open `Sentiment_Analysis.ipynb` and run the cells sequentially.

## Dataset
The project uses a publicly available sentiment analysis dataset. You may replace it with your own dataset by modifying the data preprocessing section in the notebook.

## Model Architecture
The CNN-LSTM model consists of:
- **Embedding Layer**: Converts words into dense vectors
- **CNN Layers**: Extracts local patterns from word embeddings
- **LSTM Layer**: Captures long-term dependencies in text
- **Dense Layer**: Classifies sentiment into different categories

## Model Comparison
In addition to the CNN-LSTM model, the project compares its performance with other traditional machine learning models, including:
- **Naive Bayes Classifier**: A probabilistic classifier based on Bayes' theorem.
- **Support Vector Machine (SVM)**: A powerful classification algorithm for text data.
- **Multinomial Naive Bayes (MNB)**: A variant of Naive Bayes commonly used for text classification.

The CNN-LSTM model generally achieves better results in capturing complex sentiment patterns compared to these traditional models, which rely on handcrafted features and simpler statistical approaches.

## Usage
- Modify `Sentiment_Analysis.ipynb` to experiment with different hyperparameters.
- Train the model using your dataset.
- Evaluate the model performance using accuracy, precision, recall, etc.

## Results
After training, the model achieves promising results on sentiment classification tasks. You can visualize model performance using accuracy/loss plots in the notebook.

## Technologies Used
- Python
- TensorFlow/Keras
- NLTK
- Scikit-learn
- NumPy, Pandas
- Matplotlib
- Jupyter Notebook

## Contributing
Feel free to open issues or submit pull requests for improvements!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
🚀 Happy Coding!
