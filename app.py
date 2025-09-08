import gradio as gr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import euclidean_distances

data = pd.read_csv("scaled_dataset.csv")
data.drop("CUST_ID", axis=1, inplace=True)

selected_features = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT"
]

X = data[selected_features].values

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

hierarchical_model = AgglomerativeClustering(n_clusters=4).fit(X_scaled)
dbscan_model = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)


dbscan_core_samples = dbscan_model.components_
dbscan_core_labels = dbscan_model.labels_[dbscan_model.core_sample_indices_]


def predict_cluster(username, password, algorithm, k, *features):
    
    if username != "kavin" or password != "1234":
        return "❌ Invalid login. Please try again."

    features_scaled = scaler.transform([features])

    if algorithm == "KMeans":
        model = KMeans(n_clusters=int(k), random_state=42).fit(X_scaled)
        cluster = model.predict(features_scaled)[0]
        return f"✅ This data point belongs to **Cluster {cluster}** (KMeans, k={k})"

    elif algorithm == "Hierarchical":
        new_data = np.vstack([X_scaled, features_scaled])
        labels = AgglomerativeClustering(n_clusters=4).fit_predict(new_data)
        cluster = labels[-1]
        return f"✅ This data point belongs to **Cluster {cluster}** (Hierarchical)"

    elif algorithm == "DBSCAN":
        # Compute distance to core points
        dists = euclidean_distances(features_scaled, dbscan_core_samples)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[0, nearest_idx]
        cluster = dbscan_core_labels[nearest_idx]

        if nearest_dist <= dbscan_model.eps:
            return f"✅ This data point belongs to **Cluster {cluster}** (DBSCAN, dist={nearest_dist:.2f})"
        else:
            return f"🚨 This point is considered an OUTLIER (noise) by DBSCAN."

with gr.Blocks() as demo:
    with gr.Tab("🔑 Login & Cluster Prediction"):
        gr.Markdown("## Login to Use Clustering App")

        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        algorithm = gr.Dropdown(["KMeans", "Hierarchical", "DBSCAN"], label="Select Algorithm")

        k_value = gr.Number(label="Number of Clusters (only for KMeans)", value=3)

        inputs = []
        with gr.Accordion("Enter Feature Values", open=True):
            for feature in selected_features:
                default_val = float(data[feature].median())
                inputs.append(gr.Number(label=feature, value=default_val))

        btn = gr.Button("Find Cluster")
        output = gr.Textbox(label="Result")

        btn.click(
            fn=predict_cluster,
            inputs=[username, password, algorithm, k_value] + inputs,
            outputs=output
        )

if __name__ == "__main__":
    demo.launch(share=True)
