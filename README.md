# ✍️ Handwritten Character Recognition

A deep learning project that recognizes handwritten characters (A–Z and digits 0–9) using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

> **CodeAlpha Internship Project** — Machine Learning Track

---

## 📌 Overview

This project trains a CNN model to classify handwritten characters from images. It leverages the **EMNIST** or **A-Z Handwritten Characters** dataset and achieves high accuracy by using multiple convolutional and pooling layers followed by fully connected layers.

---

## 🗂️ Project Structure

```
CodeAlpha_HandWrittenCharacterRecognition/
│
├── HandWrittenCharacterRecognition.ipynb  # Main Jupyter Notebook
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
├── model/
│   └── character_model.h5                  # Saved trained model (generated after training)
└── data/
    └── A_Z Handwritten Data.csv            # Dataset (or EMNIST loaded via library)
```

---

## 📊 Dataset

This project uses one of the following datasets:

- **A–Z Handwritten Characters Dataset** (available on Kaggle)
  - ~370,000 images of handwritten English alphabets (A–Z)
  - Each image is 28×28 pixels in grayscale
- **EMNIST Dataset** (Extended MNIST — letters/digits)
  - Loaded directly via `tensorflow_datasets` or `emnist` library

---

## 🧠 Model Architecture

The CNN model consists of:

| Layer | Details |
|---|---|
| Conv2D + ReLU | 32 filters, 3×3 kernel |
| MaxPooling2D | 2×2 pool size |
| Conv2D + ReLU | 64 filters, 3×3 kernel |
| MaxPooling2D | 2×2 pool size |
| Flatten | — |
| Dense + ReLU | 128 units |
| Dropout | Rate: 0.5 |
| Dense + Softmax | 26 units (A–Z) or 36 units (A–Z + 0–9) |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/divyadharshinidd0211-lgtm/CodeAlpha_HandWrittenCharacterRecognition.git
cd CodeAlpha_HandWrittenCharacterRecognition
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Jupyter Notebook

```bash
jupyter notebook HandWrittenCharacterRecognition.ipynb
```

Follow the cells in order to:
1. Load and preprocess the dataset
2. Build and compile the CNN model
3. Train the model
4. Evaluate accuracy on the test set
5. Predict on custom handwritten character images

---

## 📈 Results

| Metric | Value |
|---|---|
| Training Accuracy | ~98% |
| Validation Accuracy | ~95%+ |
| Test Accuracy | ~95%+ |

> Results may vary depending on the dataset used and number of epochs.

---

## 🖼️ Sample Prediction

After training, the model can predict characters from new images:

```python
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('model/character_model.h5')
img = Image.open('sample.png').convert('L').resize((28, 28))
img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
prediction = model.predict(img_array)
print("Predicted Character:", chr(ord('A') + np.argmax(prediction)))
```

---

## 🛠️ Technologies Used

- Python 3.8+
- TensorFlow / Keras
- NumPy & Pandas
- Matplotlib & Seaborn
- scikit-learn
- OpenCV
- Jupyter Notebook

---

## 🤝 Acknowledgements

- [CodeAlpha](https://www.codealpha.tech/) — Internship Program
- [Kaggle A–Z Handwritten Dataset](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
