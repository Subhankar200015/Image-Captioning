# Image Captioning: CNN + LSTM Architecture Explained
---

## 📌 Overview

Image captioning involves generating a natural language description for a given image. The typical architecture is:

**CNN (Encoder) → LSTM (Decoder) → Caption**

---

## 📦 Architecture Components

### 1. 🧠 CNN Encoder

* The encoder is usually a **pretrained CNN** like ResNet.
* It takes an image and outputs a **feature vector**, which is a numerical summary of the image.

Example:

```python
image → ResNet → feature vector
```
---

### 2. ✏️ Vocabulary

* During preprocessing, all the words in the training captions are collected.
* A unique index is assigned to each word.

Example:

```python
{ '<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3, 'tree': 4, 'dog': 5, ... }
```

The word `"tree"` might get index `4`.

---

### 3. 📝 LSTM Decoder

* The LSTM is trained to **generate words one at a time**.
* At each time step, it predicts the **next word** given the current word and the context (including the image).
* The LSTM starts with the image feature vector and the `<start>` token.

---

## 🫮 Mapping Features to Word Indices

Let’s answer the main question:

> How does the model convert image features into a word like `"tree"` (index 4)?

### 🔄 Process:

1. **Image** → CNN → Feature vector
   Example: a 2048-dim vector.

2. **Feature vector** → LSTM (as initial hidden state)

3. **LSTM output** → passed into a **Linear layer**:

   ```python
   linear = nn.Linear(hidden_dim, vocab_size)
   output = linear(lstm_output)  # output is a vector of vocab_size
   ```

4. This produces **logits** — a score for each word in the vocabulary:

   ```python
   logits = [0.1, 0.3, 0.2, 0.05, 6.8, ...]  # size = vocab_size
   ```

5. The model picks the **highest-scoring index** — e.g., index `4` → `"tree"`

6. The word at that index is emitted as the **next word** in the caption.

---

### 🔀 Training the Model

During training:

* We give the model the **actual image** and the **ground truth caption**.
* At each time step, the model is penalized (using **CrossEntropyLoss**) if it predicts the wrong word.
* Over many examples, the model **learns to associate image features** with appropriate words.

---
