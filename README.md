# 🐧 Clustering Antarctic Penguin Species

This project tackles a real-world biological classification challenge: **Can we identify distinct penguin species based solely on physical measurements, even when species labels are missing?** Using unsupervised machine learning, specifically **K-Means clustering**, we analyze morphological data from penguins collected at Palmer Station, Antarctica. The goal is to discover natural groupings in the data that likely correspond to the three known native species: **Adelie, Chinstrap, and Gentoo**. This analysis provides researchers with a data-driven method to categorize unlabeled specimens.

This project was completed using DataCamp’s Datalab environment.

---

## 🎯 Project Objectives

- Apply **unsupervised learning** to discover hidden patterns in biological data.
- Use **K-Means clustering** to group penguins based on physical characteristics.
- Determine the **optimal number of clusters (k)** using the **Elbow Method**.
- **Preprocess data** by scaling numerical features and encoding categorical variables.
- **Visualize high-dimensional clusters** using **t-SNE** for dimensionality reduction.
- Create a **summary statistics table** for each identified cluster to aid biological interpretation.

---

## 🗃️ Dataset Overview

The data comes from the `penguins.csv` file, collected by Dr. Kristen Gorman and the Palmer Station LTER.

| File | Description |
|------|-------------|
| [`penguins.csv`](./penguins.csv) | Physical measurements of individual penguins |

### Key Columns

| Column | Description |
|--------|-------------|
| `culmen_length_mm` | Length of the dorsal ridge of a bird’s beak |
| `culmen_depth_mm` | Depth of the beak |
| `flipper_length_mm` | Length of the flipper |
| `body_mass_g` | Body mass of the penguin |
| `sex` | Biological sex of the penguin (categorical) |

---

## 🔍 Methodology: 4-Step Approach

### 1. Data Preprocessing

Loaded the dataset and inspected for structure. Converted the categorical `sex` column into dummy variables and scaled all numerical features using `StandardScaler` to ensure equal weighting in the clustering algorithm.

```python
penguins_df = pd.get_dummies(penguins_df, "sex", drop_first=False)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(penguins_df)
```

### 2. Determining Optimal Clusters (Elbow Method)

Ran K-Means for k values from 1 to 9 and plotted the inertia (within-cluster sum of squares). The "elbow" point, where the rate of decrease sharply shifts, indicates the optimal k.

```python
n_cluster_inertia = {}
for n_of_clust in range(1,10):
    model = KMeans(n_clusters=n_of_clust, random_state=1)
    model.fit(X_scaled)
    n_cluster_inertia[n_of_clust] = model.inertia_
```

**Finding:** The elbow clearly occurs at **k=3**, which aligns perfectly with the known number of penguin species in the region.

### 3. Running K-Means and Visualization

Executed the final K-Means model with `n_clusters=3`. Since the data has more than 2 dimensions, used **t-SNE** to reduce it to 2D for visualization, coloring points by their assigned cluster.

```python
best_model = KMeans(n_clusters=3, random_state=1)
species = best_model.fit_predict(X_scaled)

model = TSNE(learning_rate=100, random_state=1)
transformed = model.fit_transform(X_scaled)
plt.scatter(transformed[:,0], transformed[:,1], c=species)
```

**Finding:** The t-SNE plot revealed **three distinct, well-separated clusters**, validating the choice of k=3.

### 4. Creating Final Output

Added the cluster labels as a new `"species"` column to the original DataFrame and generated a summary table showing the mean value of each feature for each cluster. This table allows researchers to characterize each group (e.g., Cluster 2 has the highest mean body mass and flipper length, likely corresponding to Gentoo penguins).

```python
penguins_df["species"] = species
stat_penguins = penguins_df.groupby("species").mean().reset_index()
```

---

## 📊 Key Findings

- ✅ **Optimal Clusters:** **3** (confirmed by Elbow Method and biological knowledge).
- 📌 **Cluster Separation:** t-SNE visualization shows **three clear, distinct groups**.
- 🧪 **Practical Output:** A statistical summary DataFrame (`stat_penguins`) that provides mean measurements for each cluster, enabling researchers to infer species characteristics.
- ✅ The unsupervised approach successfully rediscovered the known biological taxonomy from raw measurements.

---

## 🛠️ Tools Used

- **Python**
- **pandas** – for data loading, manipulation, and dummy variable creation
- **scikit-learn** – `KMeans` for clustering, `StandardScaler` for preprocessing, `TSNE` for visualization
- **Matplotlib** – for plotting the Elbow curve and cluster visualization
- **Jupyter Notebook / DataCamp Datalab** – for analysis and reporting

---

## 📌 How to Use

This project was completed in **DataCamp’s Datalab environment**. To reproduce:

1. Upload `penguins.csv` to your workspace.
2. Open the notebook.
3. Run all cells to reproduce the full analysis, from preprocessing to the final statistical summary.

> 🔗 This project was created by **DataCamp** as part of their machine learning curriculum. You can find the original exercise on their platform.

---

## ✍️ Author

Completed by **Achraf Salimi** — applying unsupervised machine learning to solve practical problems in biology and ecology.
