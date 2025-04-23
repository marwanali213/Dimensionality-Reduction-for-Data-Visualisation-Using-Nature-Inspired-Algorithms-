# Traditional-Dimenionality-reduction-techniques-for-data-visualization

### *Draft for me*

Traditional techniques for dimensionality reduction are methods used to reduce the number of input variables (or features) in a dataset while preserving as much information as possible. Here are some of the most commonly used traditional techniques:

---

### 1. **Principal Component Analysis (PCA)**
- **Type**: Linear
- **Description**: Transforms data into a new coordinate system such that the greatest variance lies on the first axis (principal component), the second greatest on the second axis, and so on.
- **Use Case**: When features are correlated and you want to reduce redundancy.

---

### 2. **Linear Discriminant Analysis (LDA)**
- **Type**: Supervised, linear
- **Description**: Projects data in a way that maximizes class separability. It tries to maximize the ratio of between-class variance to the within-class variance.
- **Use Case**: When you have labeled data and want to preserve class-discriminatory information.

---

### 3. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
- **Type**: Non-linear
- **Description**: Focuses on preserving local structure and visualizing high-dimensional data in 2 or 3 dimensions.
- **Use Case**: Data visualization, especially with clustering or manifolds.

---

### 4. **Multidimensional Scaling (MDS)**
- **Type**: Non-linear
- **Description**: Preserves the pairwise distances between data points in the reduced space.
- **Use Case**: Useful when distance between points is meaningful (e.g., similarity or dissimilarity matrices).

---

### 5. **Isomap**
- **Type**: Non-linear
- **Description**: Extends MDS by preserving geodesic distances on a manifold rather than direct Euclidean distances.
- **Use Case**: When the data lies on a non-linear manifold (e.g., Swiss roll).

---

### 6. **Independent Component Analysis (ICA)**
- **Type**: Linear
- **Description**: Assumes that the observed data are mixtures of statistically independent components and tries to separate them.
- **Use Case**: Signal processing (e.g., separating voices from background noise).

---

### 7. **Autoencoders** (Though these are neural-based, they're often considered when comparing dimensionality reduction)
- **Type**: Non-linear, unsupervised
- **Description**: Neural networks trained to reproduce the input at the output via a compressed latent representation.
- **Use Case**: Complex, non-linear feature reduction, especially in deep learning contexts.

---
