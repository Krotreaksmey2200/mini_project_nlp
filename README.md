# Khmer NLP Text Generator

A Python-based natural language processing (NLP) project that implements a text generation system for the Khmer language. The system uses an N-gram model (up to 4-grams) with a Backoff strategy to generate coherent Khmer text based on a user-provided seed.

## Features

- **Corpus Loading**: Import large text files to train the model.
- **N-gram Training**: Builds unigrams, bigrams, trigrams, and quadgrams from the loaded corpus.
- **Stupid Backoff Strategy**: Handles unseen word sequences by falling back to lower-order n-grams for probability estimation.
- **GUI Interface**: Easy-to-use graphical interface built with Tkinter.
- **Model Persistence**: Save and load trained models using `pickle` for faster startup.
- **Interpolated Smoothing**: Uses Add-k smoothing and linear interpolation for improved text generation quality.
- **Web Scraping**: Includes Jupyter Notebooks for scraping Khmer text from Wikipedia to build the corpus.

## Project Structure

- `main.py`: The primary application script with the Tkinter GUI.
- `generate_model.py`: Utility script to pre-train and save the model to a file.
- `interpolated_model.pkl`: Pre-trained model data (generated from corpus).
- `mini_project.ipynb`: Notebook for web scraping and data collection.
- `corpus.ipynb`: Notebook for corpus processing and analysis.
- `corpus.txt`: The primary training dataset.
- `wikipedia.txt`: Scraped raw text from Wikipedia.

## Requirements

- Python 3.x
- Tkinter (usually comes pre-installed with Python)
- `pickle` (standard library)
- `requests` and `beautifulsoup4` (for scraping via notebooks)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd mini_project_nlp
   ```

2. (Optional) Generate the model file:
   ```bash
   python generate_model.py
   ```

## Usage

1. **Run the Application**:
   ```bash
   python main.py
   ```

2. **Using a Pre-trained Model**:
   - Click "Load Pickled Model" to load `interpolated_model.pkl` instantly.
3. **Training from Scratch**:
   - Click "Load Corpus" and select `corpus.txt`.
   - Click "Train Model".
4. **Generate Text**:
   - Enter at least 3 Khmer words as a seed in the input field.
   - Click "Generate Text" to see the model's output.

## Technical Details

The generator uses a **Stupid Backoff** algorithm:
1. It first checks for the 4-gram probability.
2. If the 4-gram is not found, it "backs off" to the trigram.
3. If the trigram is not found, it backs off to the bigram.
4. Finally, it falls back to unigram frequency or a base probability.

This ensures that the model can still generate text even when it encounters sequences not explicitly present in the training data.

