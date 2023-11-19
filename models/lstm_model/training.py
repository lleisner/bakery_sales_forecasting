import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_lstm(model: tf.keras.models.Model, num_epochs: int, train_gen, test_gen) -> tf.keras.models.Model:
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
            # Training phase
            for X_batch, y_batch in train_gen:
                train_loss = model.train_on_batch(X_batch, y_batch)
                train_losses.append(train_loss)

            # Testing phase
            for X_batch, y_batch in test_gen:
                test_loss = model.test_on_batch(X_batch, y_batch)
                test_losses.append(test_loss)

            # Calculate and print the mean loss for the epoch if there is data
            if train_losses:
                mean_train_loss = sum(train_losses) / len(train_losses)
                print(f'Epoch {epoch+1} - Mean Train Loss: {mean_train_loss}')
            
            if test_losses:
                mean_test_loss = sum(test_losses) / len(test_losses)
                print(f'Epoch {epoch+1} - Mean Test Loss: {mean_test_loss}')

    return train_losses, test_losses
 
def custom_train_lstm(model, num_epochs, train_gen, test_gen):
    custom_history = {
        'loss': [],  # Training loss
        'val_loss': [],  # Testing loss (validation loss)
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase with progress bar
        train_loss = 0.0
        num_train_steps = 0
        progress_bar = tqdm(train_gen, unit="batch", desc="Training")
        for X_batch, y_batch in progress_bar:
            num_train_steps += 1
            train_loss += model.train_on_batch(X_batch, y_batch)
            progress_bar.set_postfix({"train_loss": train_loss / num_train_steps}, refresh=False)

        mean_train_loss = train_loss / (num_train_steps + 1)
        custom_history['loss'].append(mean_train_loss)
        print(f"\nEpoch {epoch + 1} - Mean Train Loss: {mean_train_loss:.4f}")

        # Testing phase with progress bar
        test_loss = 0.0
        num_test_steps = 0
        progress_bar = tqdm(test_gen, unit="batch", desc="Testing")
        for X_batch, y_batch in progress_bar:
            num_test_steps += 1
            test_loss += model.test_on_batch(X_batch, y_batch)
            progress_bar.set_postfix({"test_loss": test_loss / num_test_steps}, refresh=False)

        mean_test_loss = test_loss / (num_test_steps + 1)
        custom_history['val_loss'].append(mean_test_loss)
        print(f"\nEpoch {epoch + 1} - Mean Test Loss: {mean_test_loss:.4f}")

    return custom_history


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
    