import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plot training and validation loss from model.fit() history.

    Args:
    - history: The training history object returned by model.fit().
    """

    loss = history.history['loss']
    val_loss = history.history.get('val_loss', None)

    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    
    if val_loss:
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    