# 📊 Customer Clustering App

This is a simple clustering application built using **Gradio**, which allows you to cluster customer credit card data using three algorithms:
- **KMeans**
- **Hierarchical Clustering**
- **DBSCAN**

---

## ✅ Features

- 🔐 **Secure login system** (Username: `kavin`, Password: `1234`).
- 🚀 Supports multiple clustering algorithms.
- 📈 Easy-to-use interface to input feature values.
- 🎯 Shows predicted cluster or indicates outlier status.
- 📊 Visualizes results with scatter plots (optional extension).

---

## ⚙️ How to Run

### 1️⃣ Install dependencies:

pip install gradio pandas numpy scikit-learn matplotlib

---
## 🌐 Access the App

👉 Open your browser and visit:  
[https://huggingface.co/spaces/kavin2906/cluster_creditcard](https://huggingface.co/spaces/kavin2906/cluster_creditcard)

---

## 🧑‍💻 Login Credentials

- ✅ Username: `kavin`
- ✅ Password: `1234`

---

## 📝 Notes

- After successful login, the feature input fields and algorithm selector will appear.
- KMeans requires number of clusters (`k`) input.
- DBSCAN returns an OUTLIER message if the data point is too far from clusters.
- Designed for clustering customer credit card usage data.

---

## 🚀 Next Steps (Optional)

- Add functionality to export predictions as CSV.
- Deploy online using platforms like Hugging Face Spaces.
- Add graphical cluster visualization for each algorithm.

---

# Clustering Analysis Results

This project performs clustering on a dataset using three different algorithms: **KMeans**, **Hierarchical Clustering**, and **DBSCAN**. Below are the results comparing the number of clusters and the Silhouette Score for each algorithm.

| Algorithm    | Number of Clusters | Silhouette Score | Notes                                     |
| ----------- | ------------------ | --------------- | --------------------------------------      |
| KMeans      | 2                  | 0.47            | Best score among the three                  |
| Hierarchical| 2                  | 0.39            | Slightly lower than KMeans                  |
| DBSCAN      | 4                  | 0.23            | Slightly lower than KMeans and Hierarchical |

---

## Summary

- **KMeans** achieved the best silhouette score of **0.47** with **2 clusters**.
- **Hierarchical Clustering** performed reasonably well with a silhouette score of **0.39**.
- **DBSCAN** identified **4 clusters**, but with a lower silhouette score of **0.23**.

---

## Conclusion

Based on the silhouette scores, **KMeans with 2 clusters** is the most appropriate clustering method for this dataset.

---

**AUTHOR:KAVINKUMAR T**

---
