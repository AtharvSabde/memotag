import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Load the dataset - replace with your file path
file_path = r"C:\Users\Admin\Desktop\memotag_test\transcribe_1\extracted_features.csv"  # You can update this path
df = pd.read_csv(file_path)
print(f"Loaded {len(df)} speech samples with {len(df.columns)-1} features")

# Store sample_id separately for reference
sample_ids = df["sample_id"].copy()
feature_names = [col for col in df.columns if col != "sample_id"]
X = df.drop(columns=["sample_id"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Run Isolation Forest for anomaly detection
print("Running Isolation Forest anomaly detection...")
iso_forest = IsolationForest(contamination=0.3, random_state=42)
anomaly_scores = iso_forest.fit_predict(X_scaled)
anomaly_decision = ["Anomaly" if score == -1 else "Normal" for score in anomaly_scores]

# Create a DataFrame with results
results_df = pd.DataFrame({
    'sample_id': sample_ids,
    'isolation_forest': anomaly_decision,
    'anomaly_score': iso_forest.decision_function(X_scaled) * -1  # Higher = more anomalous
})

# Add original features back for analysis
for i, feature in enumerate(feature_names):
    results_df[feature] = X.iloc[:, i]

# Calculate correlation between each feature and the anomaly score
print("\n=== ANALYZING FEATURE IMPORTANCE USING CORRELATION WITH ANOMALY SCORE ===")
correlation_with_anomaly = {}

for feature in feature_names:
    # Use Spearman correlation as it works better for non-linear relationships
    correlation, p_value = spearmanr(results_df[feature], results_df['anomaly_score'])
    correlation_with_anomaly[feature] = {
        'correlation': abs(correlation),  # Take absolute value to measure strength
        'direction': 'positive' if correlation > 0 else 'negative',
        'p_value': p_value
    }

# Sort features by strength of correlation
sorted_correlations = sorted(correlation_with_anomaly.items(), 
                            key=lambda x: x[1]['correlation'], 
                            reverse=True)

print("Top 10 features most strongly correlated with anomaly detection:")
for i, (feature, stats) in enumerate(sorted_correlations[:10]):
    print(f"{i+1}. {feature}: {stats['correlation']:.4f} ({stats['direction']} correlation, p={stats['p_value']:.4f})")

# Create DataFrame for plotting feature importance
importance_df = pd.DataFrame({
    'Feature': [feature for feature, _ in sorted_correlations],
    'Correlation': [stats['correlation'] for _, stats in sorted_correlations]
}).sort_values('Correlation', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_df.head(10), x='Correlation', y='Feature')
plt.title('Top 10 Most Important Features for Anomaly Detection')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# Apply PCA for visualization
pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_scaled)

# Add PCA components to results dataframe
results_df['PCA1'] = pca_results[:, 0]
results_df['PCA2'] = pca_results[:, 1]

# Create PCA visualization of anomalies
plt.figure(figsize=(10, 8))
colors = {'Anomaly': 'red', 'Normal': 'blue'}

# Plot Isolation Forest results
sns.scatterplot(
    x='PCA1', 
    y='PCA2', 
    hue='isolation_forest',
    palette=colors,
    data=results_df,
    s=100  # Size of points
)

plt.title('Isolation Forest Anomalies', fontsize=14)
plt.xlabel('PCA1', fontsize=12)
plt.ylabel('PCA2', fontsize=12)
plt.legend(title='isolation_forest')
plt.tight_layout()

# Save only the PCA visualization
plt.savefig('isolation_forest_pca_visualization.png', dpi=300)
plt.show()

# Save results to CSV
results_df.to_csv('cognitive_stress_detection_results.csv', index=False)
print(f"Detection results saved to 'cognitive_stress_detection_results.csv'")
print(f"Feature importance visualization saved to 'feature_importance.png'")
print(f"PCA visualization of Isolation Forest results saved to 'isolation_forest_pca_visualization.png'")