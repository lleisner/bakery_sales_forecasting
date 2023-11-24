import tensorflow as tf

def custom_time_series_loss(future_steps, length_of_day):
    """
    Custom Time Series Loss Function

    Calculates a composite loss for time series data, considering both individual hours and daily aggregation.

    Parameters:
    - future_steps (int): The number of hours from the end of each sequence to consider for loss calculation.
    - length_of_day (int): The number of hours that make up a complete day in the time series.

    Returns:
    - loss_function: A TensorFlow loss function that combines Mean Squared Error (MSE) for individual hours and
    MSE for daily aggregations to compute the total loss.

    Usage:
    This loss function is designed for time series data where the goal is to predict future hours. The `future_steps` parameter
    specifies how many recent hours are considered, and `length_of_day` indicates the number of hours in a complete day.
    The function calculates the MSE for the individual hours in the last `n` steps and the MSE for the sums of values
    for entire days. These losses are combined to produce the total loss used during model training.
    """
    def loss_function(y_true, y_pred):


        # Calculate the sum of values for entire days in the last n steps
        y_true_day_sum = tf.reduce_sum(tf.reshape(y_true, (-1, length_of_day)), axis=1)
        y_pred_day_sum = tf.reduce_sum(tf.reshape(y_pred, (-1, length_of_day)), axis=1)

        # Compute the mean squared error for the last n elements with weights
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Weigh the sum of the whole day to avoid overwhelming the hourly loss
        daily_weight = 1/length_of_day

        # Calculate the MSE between the sum of values for entire days with weights
        day_mse = tf.reduce_mean(daily_weight * tf.square(y_true_day_sum - y_pred_day_sum))

        # Combine both losses
        total_loss = mse + day_mse

        return total_loss

    return loss_function



class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
    
    def call(self, y_true, y_pred):
        y_true_sum = tf.reduce_sum(tf.reshape(y_true, (-1, self.interval)), axis=1)
        y_pred_sum = tf.reduce_sum(tf.reshape(y_pred, (-1, self.interval)), axis=1)

        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        weight = 1/self.interval

        summed_mse = tf.reduce_mean(daily_weight * tf.square(y_true_sum - y_pred_sum))

        total_loss =  mse + summed_mse

        return total_loss
