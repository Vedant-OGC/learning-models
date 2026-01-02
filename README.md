# 📊 Machine Learning Visual Experiments

This repository contains a collection of **machine learning experiments** built to understand ML concepts **visually and intuitively**, rather than treating them as black boxes.

The goal is simple:  
> *If you can visualize it, you can truly understand it.*

---

## 🔍 What’s inside?

### 1️⃣ Linear Regression (From Scratch → Intuition)
A basic regression model that learns the relationship between **experience and salary**.

**Concepts covered:**
- Model fitting using **least squares**
- Coefficients & intercept interpretation
- Train-test split
- Error measurement using **Mean Squared Error (MSE)**

📂 Folder: `LR 1/`  
📈 Includes dataset, training scripts, predictions, and visualizations.

---

### 2️⃣ Gradient Descent Visualization
Instead of treating optimization as magic, this script visualizes how **gradient descent** actually works.

**Concepts covered:**
- Loss surface
- Iterative parameter updates
- Effect of learning rate
- Convergence toward a minimum

📄 File: `gradient_descent.py`

This helps build intuition around **how models learn**, not just what libraries do.

---

### 3️⃣ High-Dimensional Data Visualization (PCA & t-SNE)
Modern ML often works in **high-dimensional feature spaces**.  
This experiment shows how to *see* such data.

**Pipeline:**
- Generate high-dimensional data (64D+)
- Reduce dimensions using **PCA** (variance-preserving)
- Apply **t-SNE** to reveal nonlinear cluster structures in 2D

📄 File: `tsne_pca.py`

This demonstrates how embeddings behave geometrically in lower dimensions.

---

## 🛠️ Tech Stack
- **Python**
- NumPy, Pandas
- Matplotlib
- scikit-learn

All scripts are lightweight, readable, and beginner-friendly.

---

## 🚀 How to run

1. Clone the repository:
```bash
git clone https://github.com/Vedant-OGC/learning-models.git
cd learning-models
