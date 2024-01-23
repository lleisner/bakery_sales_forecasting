from datetime import datetime, timedelta

class Settings:
    def __init__(self):
        self.past_days = 64     # 64
        self.future_days = 4
        self.length_of_day = 12
        
        self.strides = 12
        
        self.future_steps = self.future_days * self.length_of_day
        self.seq_length = self.past_days * self.length_of_day

        self.num_epochs = 200
        self.early_stopping_patience = max(self.num_epochs//10, 2)
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.validation_size = 0.25
        self.test_size = 0.05
        self.iters_per_epoch = 2
        

    def calculate_steps(self, len_data):
        steps_per_epoch = self.iters_per_epoch * ((len_data // self.strides) * (1-(self.test_size + self.validation_size)) // self.batch_size - (self.seq_length/self.strides)//self.batch_size)
        validation_steps = max((len_data // self.strides) * self.validation_size // self.batch_size, 1) 
        test_steps = max((len_data // self.strides) * self.test_size // self.batch_size, 1) 
        return steps_per_epoch, validation_steps, test_steps


class ProviderConfigs:
    def __init__(self):
        self.start_date = '2019-01-01'
        #self.end_date = str(datetime.combine(datetime.now() + timedelta(days=1), datetime.min.time()).date())
        self.end_date = '2023-08-01'
        self.start_time = '06:00:00'
        self.end_time = '17:00:00'
        self.item_selection = ["broetchen", "plunder"]

class ProcessorConfigs:
    def __init__(self, settings):
        self.covariate_selection = ["datetime", "is_open", "ferien", "fahrten", "weather"]  
        self.reduce_one_hots = False 
        self.future_days = settings.future_days     
        self.temp_encoder = "standard"
        self.def_encoder = "standard"

class PipelineConfigs:
    def __init__(self, settings, num_features, num_targets):
        self.window_size = settings.seq_length
        self.sliding_step = settings.strides

        self.num_epochs = settings.num_epochs
        self.num_repeats = settings.num_epochs * settings.iters_per_epoch
        self.batch_size = settings.batch_size
        self.validation_size = settings.validation_size
        self.test_size = settings.test_size

        self.num_targets = num_targets
        self.num_features = num_features

        self.buffer_size = 10000
        self.seed = 42

class TransformerConfigs:
    def __init__(self, settings, num_features, num_targets):
        self.seq_len = settings.seq_length
        self.pred_len = settings.future_steps
        self.num_targets = num_targets
        self.num_features = num_features
        
        self.output_attention = False
        self.dropout = 0.3  # 0.4
        self.d_model = 32  # 16
        self.n_heads = 8
        self.d_ff = 128    # 64
        self.activation = 'gelu'
        self.e_layers = 2
        self.clip = 2.0
        self.use_amp = True
        self.use_norm = True
