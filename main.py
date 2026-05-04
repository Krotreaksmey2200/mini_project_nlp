import tkinter as tk
from tkinter import filedialog
import random
import pickle
from collections import Counter
import os

# Global variables
tokens = []
vocab = set()
count_4, count_3, count_2, count_1 = None, None, None, None
token_count = 0

# ---------- Load Corpus ----------
def load_file():
    global tokens
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    tokens = text.split()
    output_box.insert(tk.END, "Corpus loaded!\n")

# ---------- Load Pickled Model ----------
def load_pickled_model():
    global vocab, count_4, count_3, count_2, count_1, token_count
    file_path = "interpolated_model.pkl"
    if not os.path.exists(file_path):
        output_box.insert(tk.END, f"Error: {file_path} not found. Please train a model first.\n")
        return

    try:
        with open(file_path, "rb") as f:
            model_data = pickle.load(f)
        
        vocab = model_data["vocab"]
        count_4 = model_data["count_4"]
        count_3 = model_data["count_3"]
        count_2 = model_data["count_2"]
        count_1 = model_data["count_1"]
        token_count = model_data["token_count"]
        
        output_box.insert(tk.END, f"Model loaded from {file_path} successfully!\n")
    except Exception as e:
        output_box.insert(tk.END, f"Error loading model: {e}\n")

# ---------- Train Model ----------
def train_model():
    global vocab, count_4, count_3, count_2, count_1, token_count
    
    if not tokens:
        output_box.insert(tk.END, "Please load a corpus first!\n")
        return

    def build_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    vocab = set(tokens)
    token_count = len(tokens)
    
    count_4 = Counter(build_ngrams(tokens, 4))
    count_3 = Counter(build_ngrams(tokens, 3))
    count_2 = Counter(build_ngrams(tokens, 2))
    count_1 = Counter(tokens)
    
    # Save the model after training
    model_data = {
        "vocab": vocab,
        "count_4": count_4,
        "count_3": count_3,
        "count_2": count_2,
        "count_1": count_1,
        "token_count": token_count
    }
    with open("interpolated_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
        
    output_box.insert(tk.END, "Model trained and saved to interpolated_model.pkl!\n")

# ---------- Smoothed Probability ----------
def smoothed_prob(count_ngram, count_prev, V, k=0.01):
    return (count_ngram + k) / (count_prev + k*V)

# ---------- Interpolated Model ----------
def interpolated_prob(w1, w2, w3, w4):
    V = len(vocab)
    if V == 0: return 0
    
    lambda1, lambda2, lambda3, lambda4 = 0.25, 0.25, 0.25, 0.25
    
    p4 = smoothed_prob(count_4[(w1,w2,w3,w4)], count_3[(w1,w2,w3)], V)
    p3 = smoothed_prob(count_3[(w2,w3,w4)], count_2[(w2,w3)], V)
    p2 = smoothed_prob(count_2[(w3,w4)], count_1[w3], V)
    p1 = smoothed_prob(count_1[w4], token_count, V)
    
    return lambda1*p4 + lambda2*p3 + lambda3*p2 + lambda4*p1

# ---------- Generate Text ----------
def generate_text():
    if not vocab:
        output_box.insert(tk.END, "Model not trained or loaded!\n")
        return
        
    seed_text = seed_entry.get().strip()
    seed = seed_text.split()
    
    if len(seed) < 3:
        output_box.insert(tk.END, "Enter at least 3 words!\n")
        return
    
    result = seed[:]
    
    output_box.insert(tk.END, "Generating text...\n")
    root.update_idletasks() # Refresh GUI
    
    for _ in range(20):
        w1,w2,w3 = result[-3:]
        candidates = list(vocab)
        
        # Calculate probabilities for all candidates
        probs = [interpolated_prob(w1,w2,w3,w) for w in candidates]
        
        # Safety check for sum of probabilities
        prob_sum = sum(probs)
        if prob_sum == 0:
            probs = [1/len(candidates)] * len(candidates)
        
        next_word = random.choices(candidates, weights=probs)[0]
        result.append(next_word)
    
    output_box.insert(tk.END, "Generated:\n" + " ".join(result) + "\n\n")

# ---------- GUI ----------
root = tk.Tk()
root.title("Khmer NLP Text Generator (Interpolated)")

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="Load Corpus", command=load_file).grid(row=0, column=0, padx=5)
tk.Button(frame, text="Train & Save Model", command=train_model).grid(row=0, column=1, padx=5)
tk.Button(frame, text="Load Pickled Model", command=load_pickled_model).grid(row=0, column=2, padx=5)

tk.Label(root, text="Enter Seed (at least 3 words):").pack()
seed_entry = tk.Entry(root, width=50)
seed_entry.pack()

tk.Button(root, text="Generate Text", command=generate_text).pack(pady=5)

output_box = tk.Text(root, height=15, width=60)
output_box.pack(padx=10, pady=10)

# Auto-load the model if it exists
root.after(100, load_pickled_model)

root.mainloop()
