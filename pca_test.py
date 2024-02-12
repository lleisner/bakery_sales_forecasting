import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.configs import Settings, ProcessorConfigs, ProviderConfigs
from data_provider.data_provider import DataProvider
from data_provider.data_encoder import DataProcessor

settings = Settings()
configs = ProviderConfigs()
provider = DataProvider(configs)
provider.create_new_database(provider_list=['sales'])
df = provider.load_database()
p_configs = ProcessorConfigs(settings)

p_configs.reduce_one_hots = True
p_configs.create_sales_features = False
processor = DataProcessor(data=df, configs=p_configs)

encoding = processor.fit_and_encode()
#encoding.drop(columns=["sales"], inplace=True)
print(encoding)

pca = PCA(n_components=None)  # Replace `None` with the number of components you want to keep

# Fit and transform the scaled data
pca_features = pca.fit_transform(encoding)

explained_variance = pca.explained_variance_ratio_

# This gives you an array of the variance explained by each principal component
print(explained_variance)

# Cumulative explained variance can also be insightful
cumulative_explained_variance = np.cumsum(explained_variance)
print(cumulative_explained_variance)

n_components_to_keep = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1

print(n_components_to_keep)

pca = PCA(n_components=n_components_to_keep)
final_features = pca.fit_transform(encoding)

print(final_features)

def plot_scree(pca):
    explained_variance = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    

def plot_biplot(pca, pca_features, feature_names):
    plt.figure(figsize=(10, 7))
    feature_vectors = pca.components_.T

    # Use the first two components for the biplot
    xs = pca_features[:, 0]
    ys = pca_features[:, 1]

    # Scale the principal components (optional)
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    plt.scatter(xs * scalex, ys * scaley, s=5)

    # Plot each feature vector as an arrow
    for i, v in enumerate(feature_vectors):
        # Check to ensure the loop does not go beyond the number of feature names
        if i < len(feature_names):
            plt.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.1, linewidth=2, color='red')
            plt.text(v[0]*1.2, v[1]*1.2, feature_names[i], color='black', ha='center', va='center', fontsize=9)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    plt.title("PCA Biplot")
    plt.grid(True)
    plt.show()

# Assuming `pca_features` is the PCA-transformed data and `df.columns` are the feature names
plot_biplot(pca, pca_features, df.columns)


# Assuming `pca` is your fitted PCA object
plot_scree(pca)
