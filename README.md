# Khmer NLP Text Generator (Interpolated)

A Python-based natural language processing (NLP) application for generating Khmer text. This project implements a 4-gram language model using linear interpolation and Add-k smoothing to provide coherent and context-aware text generation.

## Features

- **Automatic Model Loading**: On startup, the application automatically looks for and loads `interpolated_model.pkl`.
- **Interpolated Smoothing**: Combines probabilities from quadgrams, trigrams, bigrams, and unigrams for superior text quality compared to simple backoff.
- **Add-k Smoothing**: Handles out-of-vocabulary words and zero-frequency counts gracefully.
- **Unified Training Workflow**: "Train & Save" feature that processes the corpus and persists the model for future use in one click.
- **Interactive GUI**: User-friendly Tkinter interface with real-time status updates and generation progress.
- **Web Scraping Utilities**: Includes notebooks for expanding the training corpus from Wikipedia.

##  Project Structure

- `main.py`: The main GUI application.
- `generate_model.py`: CLI utility to pre-calculate the model from `corpus.txt`.
- `interpolated_model.pkl`: The serialized model data (binary).
- `corpus.txt`: The primary training dataset (Khmer text).
- `mini_project.ipynb` & `corpus.ipynb`: Research and scraping notebooks.

##  Requirements

- **Python 3.x**
- **Tkinter**: (Included with most Python installations)
- **Pickle**: (Standard library)
- **Requests & BeautifulSoup4**: (Only required for running scraping notebooks)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mini_project_nlp
   ```

2. **(Optional) Build the model**:
   If `interpolated_model.pkl` is missing, you can generate it from the corpus:
   ```bash
   python generate_model.py
   ```

##  Usage

1. **Start the App**:
   ```bash
   python main.py
   ```
   *The app will automatically attempt to load the pre-trained model on startup.*

2. **Using the Interface**:
   - **Load Corpus**: Select a text file to use for training.
   - **Train & Save Model**: Trains the model on the current corpus and saves it to disk.
   - **Load Pickled Model**: Manually reload the saved model file.
   - **Generate Text**: Enter at least 3 Khmer words as a seed and click generate.

## 🔬 Technical Methodology

The engine calculates the probability of the next word $w_4$ given $w_1, w_2, w_3$ using:

$$P(w_4 | w_1, w_2, w_3) = \lambda_1 \hat{P}(w_4 | w_1, w_2, w_3) + \lambda_2 \hat{P}(w_4 | w_2, w_3) + \lambda_3 \hat{P}(w_4 | w_3) + \lambda_4 \hat{P}(w_4)$$

Where:
- Each $\hat{P}$ is an **Add-k smoothed** probability.
- $\lambda$ weights are balanced (0.25 each) to ensure influence from all N-gram levels.


