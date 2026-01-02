import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


digits = load_digits()
X = digits.data        # shape: (n_samples, 64)
y = digits.target      # labels 0–9


pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X)


tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42
)
X_tsne = tsne.fit_transform(X_pca)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap="tab10",
    s=15
)

plt.colorbar(scatter, label="Digit Class")
plt.title("t-SNE Visualization of Dimensional Data")
plt.xlabel("Competent Score")
plt.ylabel("Digit Class")
plt.show()
