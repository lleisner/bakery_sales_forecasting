from datetime import datetime, timedelta
import tensorflow as tf
from utils.loss import CustomLoss, CombinedLossWithDynamicWeights
import math

class Settings:
    def __init__(self):
        self.past_days = 12   # 64
        self.future_days = 12
        self.length_of_day = 8
        
        self.strides = 1
        
        self.future_steps = self.future_days * self.length_of_day
        self.seq_length = self.past_days * self.length_of_day
        
        self.future_steps = 96
        self.seq_length = 96

        self.num_epochs = 15
        self.early_stopping_patience = max(self.num_epochs//10, 1)
        
        self.batch_size = 32
        self.validation_size = 0.2
        self.test_size = 0.1
        self.iters_per_epoch = 1
        

    def calculate_steps(self, len_data):
        steps_per_epoch = self.iters_per_epoch * ((len_data // self.strides) * (1-(self.test_size + self.validation_size)) // self.batch_size - (self.seq_length/self.strides)//self.batch_size)
        validation_steps = max((len_data // self.strides) * self.validation_size // self.batch_size, 1) 
        test_steps = max((len_data // self.strides) * self.test_size // self.batch_size, 1) 
        print(f"len_data: {len_data}, steps_per_epoch: {steps_per_epoch}, valid_steps: {validation_steps}, test_steps: {test_steps}")
        return steps_per_epoch, validation_steps, test_steps


class ProviderConfigs:
    def __init__(self):
        self.start_date = '2019-01-01'
        #self.end_date = str(datetime.combine(datetime.now() + timedelta(days=1), datetime.min.time()).date())
        self.end_date = '2023-08-01'
        self.start_time = '08:00:00'
        self.end_time = '15:00:00'
        self.item_selection = ["broetchen", "plunder"]

class ProcessorConfigs:
    def __init__(self, settings):
        self.covariate_selection =["is_open", "gaeste", "ferien", "fahrten", "weather"]#, "datetime"]  
        self.reduce_one_hots = False 
        self.create_sales_features = True
        self.future_days = settings.future_days     
        self.aggregate = True
        self.temp_encoder = "standard"
        self.def_encoder = "standard"

class PipelineConfigs:
    def __init__(self, settings):
        self.window_size = settings.seq_length
        self.sliding_step = settings.strides

        self.num_epochs = settings.num_epochs
        self.num_repeats = settings.num_epochs * settings.iters_per_epoch
        self.batch_size = settings.batch_size
        self.validation_size = settings.validation_size
        self.test_size = settings.test_size


        self.buffer_size = 10000
        self.seed = 42
        
        self.validation_window_size = self.window_size * 4

class TransformerConfigs:
    def __init__(self, settings, num_features, num_targets):
        self.seq_len = settings.seq_length
        self.pred_len = settings.future_steps
        self.num_targets = num_targets
        self.num_features = num_features
        
        self.output_attention = True
        self.dropout = 0.2  # 0.4
        self.d_model = 256  # 16
        self.n_heads = 8
        self.d_ff = 1024    # 64
        self.activation = 'gelu'
        self.e_layers = 2
        self.clip = 5.0
        self.use_amp = True
        self.use_norm = True
        

        #self.loss = CustomLoss(settings.length_of_day)
        self.loss = CombinedLossWithDynamicWeights(
            hourly_weight=0.2,
            daily_weight=0.8,
            underprediction_penalty=1.5,
            overprediction_penalty=1.0,
            interval=settings.length_of_day
        )
        self.loss = tf.keras.losses.MeanSquaredError()
        self.loss = tf.keras.losses.RootMeanSquaredError()
        #self.loss = tf.keras.losses.MeanAbsoluteError()
        #self.loss = tf.keras.losses.MeanAbsolutePercentageError()
        self.learning_rate = 0.0001
        
class TIDEConfigs:
    def __init__(self, settings, num_features, num_targets):
        self.seq_len = settings.seq_length
        self.pred_len = settings.future_steps
        self.num_targets = num_targets
        self.num_features = num_features
        self.hidden_dims = [32,32,32,32]
        self.time_encoder_dims = [64,4]
        self.dec_out = num_targets
        self.cat_size = 1
        self.transform = False
        self.cat_emb = 1
        self.layer_norm = True
        self.dropout_rate = 0.5
        


def call_on_model(model_class, model_configs):
    model = model_class(model_configs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_configs.learning_rate), loss=model_configs.loss, metrics=["mse", "mae"], weighted_metrics=[])
    return model


def build_model(hp, model, configs):
    configs.d_model = hp.Int("d_model", min_value=16, max_value=128, step=16)
    configs.d_ff = hp.Int("d_ff", min_value=128, max_value=1024, step=128)
    configs.n_heads = hp.Int("n_heads", min_value=4, max_value=8, step=4)
    configs.e_layers = hp.Int("e_layers", min_value=2, max_value=4, step=2)
    #configs.use_norm = hp.Boolean("use_norm")
    configs.dropout = hp.Float("dropout", min_value=0.2, max_value=0.2, step=0.2)
    configs.learning_rate = hp.Float("lr", min_value=0.0005, max_value=0.0005, sampling="log")
    
    return call_on_model(model, configs)
    


def get_d_model(num_targets=32, d_min=32, d_max=512):
    """
    Calculate the optimal model dimension as a clamped power of two based on the number of targets.

    Parameters:
        num_targets (int): The base number of targets to determine the dimension size. Default is 32.
        d_min (int): Minimum dimension size. Default is 16.
        d_max (int): Maximum dimension size. Default is 512.

    Returns:
        int: Dimension size constrained between d_min and d_max.
    """
    optimal = 2 ** math.ceil(math.log2(num_targets))
    return min(max(optimal, d_min), d_max)