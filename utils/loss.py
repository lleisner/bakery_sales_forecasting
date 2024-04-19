import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
    
    def call(self, y_true, y_pred):
        y_true_sum = tf.reduce_sum(tf.reshape(y_true, (-1, self.interval)), axis=1)
        y_pred_sum = tf.reduce_sum(tf.reshape(y_pred, (-1, self.interval)), axis=1)

        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        weight = 1/self.interval

        summed_mse = tf.reduce_mean(weight * tf.square(y_true_sum - y_pred_sum))

        total_loss =  mse + summed_mse

        return total_loss
    

class CombinedLossWithDynamicWeights(tf.keras.losses.Loss):
    def __init__(self, hourly_weight, daily_weight, underprediction_penalty, overprediction_penalty, interval, name="combined_loss_with_dynamic_weights"):
        super().__init__(name=name)
        self.hourly_weight = hourly_weight
        self.daily_weight = daily_weight
        self.underprediction_penalty = underprediction_penalty
        self.overprediction_penalty = overprediction_penalty
        self.interval = interval

    def call(self, y_true, y_pred):
        # Calculate sales volume based weights dynamically
        sales_volume = tf.reduce_sum(y_true, axis=[0, 1])
        
        # Assign volume based weights with logarithmic scaling to variates 
        shift_constant = tf.abs(tf.reduce_min(sales_volume)) + 1  # Ensure the minimum value after shifting is at least 1
        adjusted_sales_volume = sales_volume + shift_constant
        item_weights = tf.math.log(adjusted_sales_volume) / tf.reduce_max(tf.math.log(adjusted_sales_volume))

        
        # Hourly loss component
        hourly_error = y_true - y_pred
        hourly_underprediction_error = tf.nn.relu(hourly_error) * self.underprediction_penalty
        hourly_overprediction_error = tf.nn.relu(-hourly_error) * self.overprediction_penalty
        hourly_weighted_error = (hourly_underprediction_error + hourly_overprediction_error) * item_weights
        hourly_loss = tf.reduce_mean(tf.square(hourly_weighted_error))

        # Daily aggregated loss component
        batch_size = y_true.shape[0]
        y_true_daily = tf.reduce_sum(tf.reshape(y_true, (batch_size, -1, self.interval, y_true.shape[-1])), axis=1)
        y_pred_daily = tf.reduce_sum(tf.reshape(y_pred, (batch_size, -1, self.interval, y_pred.shape[-1])), axis=1)

        #print(y_true_daily.shape)
        daily_error = y_true_daily - y_pred_daily
        #print(daily_error)
        daily_underprediction_error = tf.nn.relu(daily_error) * self.underprediction_penalty
        daily_overprediction_error = tf.nn.relu(-daily_error) * self.overprediction_penalty
        daily_weighted_error = (daily_underprediction_error + daily_overprediction_error) * item_weights
        daily_loss = tf.reduce_mean(tf.square(daily_weighted_error))
        
        h_loss = tf.identity(hourly_loss, name="hourly_loss_value")
        d_loss = tf.identity(daily_loss, name="daily_loss_value")
        #tf.print("hourly loss value:", h_loss * self.hourly_weight)
        #tf.print("daily loss value:", d_loss * self.daily_weight)
     

        # Combine hourly and daily loss components
        combined_loss = self.hourly_weight * hourly_loss + self.daily_weight * daily_loss
        return combined_loss



class SMAPELoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='SMAPELoss'):
        """
        Initialize the SMAPELoss class.

        Args:
        reduction: Type of tf.keras.losses.Reduction to apply to loss. Default value is AUTO.
        name: Optional name for the operations created when applying the loss. Defaults to 'SMAPELoss'.
        """
        super(SMAPELoss, self).__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        """
        Calculate the SMAPE loss between `y_true` and `y_pred`.

        Args:
        y_true: Actual values. Tensor of the same shape as `y_pred`.
        y_pred: Predicted values. Tensor of the same shape as `y_true`.

        Returns:
        smape: The SMAPE loss between `y_true` and `y_pred`.
        """
        numerator = tf.abs(y_true - y_pred)
        denominator = tf.maximum(tf.abs(y_true) + tf.abs(y_pred), 1e-7)  # Avoid division by zero
        smape = 2.0 * tf.reduce_mean(numerator / denominator)
        return smape
