import matplotlib.pyplot as plt


def plot_attention_head(variate_labels, attention):
    """
    Plots a single attention head focusing on variate tokens.

    Args:
        variate_labels: A list of labels for each variate token.
        attention: A 2D numpy array of shape (num_variates, num_variates) containing
                   attention weights.
    """
    ax = plt.gca()
    cax = ax.matshow(attention, cmap='viridis')
    plt.colorbar(cax, ax=ax)

    ax.set_xticks(range(len(variate_labels)))
    ax.set_yticks(range(len(variate_labels)))

    ax.set_xticklabels(variate_labels, rotation=90)
    ax.set_yticklabels(variate_labels)

def plot_attention_weights(variate_labels, attention_heads):
    """
    Plots the attention weights across multiple heads for variate tokens.

    Args:
        variate_labels: A list of labels for each variate token.
        attention_heads: A list of 2D numpy arrays where each array represents the
                         attention weights for a particular head over variate tokens.
    """
    assert len(attention_heads) <= 8, "Can only plot up to 8 heads."
    
    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)
        plot_attention_head(variate_labels, head)
        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()