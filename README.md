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
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   (Ensure you have Python 3.7+ installed.)

3. Launch Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
4. Open `Sentiment_Analysis.ipynb` and run the cells sequentially.

## Dataset
The project uses a publicly available sentiment analysis dataset. You may replace it with your own dataset by modifying the data preprocessing section in the notebook.

## Model Architecture
The CNN-LSTM model consists of:
- **Embedding Layer**: Converts words into dense vectors
- **CNN Layers**: Extracts local patterns from word embeddings
- **LSTM Layer**: Captures long-term dependencies in text
- **Dense Layer**: Classifies sentiment into different categories

## Usage
- Modify `Sentiment_Analysis.ipynb` to experiment with different hyperparameters.
- Train the model using your dataset.
- Evaluate the model performance using accuracy, precision, recall, etc.

## Results
After training, the model achieves promising results on sentiment classification tasks. You can visualize model performance using accuracy/loss plots in the notebook.

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook

## Contributing
Feel free to open issues or submit pull requests for improvements!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
🚀 Happy Coding!
