import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(attention, title="Attention Weights"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.show()

# Select a single attention head and layer to visualize, if multiple are present
# For example, attention_weights might be a list of tensors if you have multiple layers or heads
attention_to_plot = attention_weights[0].numpy()  # Assuming you want to visualize the first head's weights

# Plot the heatmap
plot_attention_heatmap(attention_to_plot)