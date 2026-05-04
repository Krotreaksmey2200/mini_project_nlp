import tkinter as tk
from tkinter import filedialog
import random
from collections import Counter

# Global variables
tokens = []
vocab = set()
count_4, count_3, count_2, count_1 = None, None, None, None

# ---------- Load Corpus ----------
def load_file():
    global tokens
    file_path = filedialog.askopenfilename()
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    tokens = text.split()
    output_box.insert(tk.END, "Corpus loaded!\n")

# ---------- Train Model ----------
def train_model():
    global vocab, count_4, count_3, count_2, count_1
    
    def build_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    vocab = set(tokens)
    
    count_4 = Counter(build_ngrams(tokens, 4))
    count_3 = Counter(build_ngrams(tokens, 3))
    count_2 = Counter(build_ngrams(tokens, 2))
    count_1 = Counter(tokens)
    
    output_box.insert(tk.END, "Model trained!\n")

# ---------- Backoff Model ----------
def backoff_prob(w1,w2,w3,w4):
    if count_3[(w1,w2,w3)] > 0 and count_4[(w1,w2,w3,w4)] > 0:
        return count_4[(w1,w2,w3,w4)] / count_3[(w1,w2,w3)]
    elif count_2[(w2,w3)] > 0 and count_3[(w2,w3,w4)] > 0:
        return count_3[(w2,w3,w4)] / count_2[(w2,w3)]
    elif count_1[w3] > 0 and count_2[(w3,w4)] > 0:
        return count_2[(w3,w4)] / count_1[w3]
    else:
        return count_1[w4] / len(tokens)

# ---------- Generate Text ----------
def generate_text():
    seed = seed_entry.get().split()
    
    if len(seed) < 3:
        output_box.insert(tk.END, "Enter at least 3 words!\n")
        return
    
    result = seed[:]
    
    for _ in range(20):
        w1,w2,w3 = result[-3:]
        candidates = list(vocab)
        probs = [backoff_prob(w1,w2,w3,w) for w in candidates]
        
        if sum(probs) == 0:
            probs = [1]*len(probs)
        
        next_word = random.choices(candidates, weights=probs)[0]
        result.append(next_word)
    
    output_box.insert(tk.END, "Generated:\n" + " ".join(result) + "\n\n")

# ---------- GUI ----------
root = tk.Tk()
root.title("Khmer NLP Text Generator")

tk.Button(root, text="Load Corpus", command=load_file).pack()
tk.Button(root, text="Train Model", command=train_model).pack()

tk.Label(root, text="Enter Seed (3 words):").pack()
seed_entry = tk.Entry(root, width=50)
seed_entry.pack()

tk.Button(root, text="Generate Text", command=generate_text).pack()

output_box = tk.Text(root, height=15, width=60)
output_box.pack()

root.mainloop()