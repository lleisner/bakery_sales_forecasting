class Args:
    def __init__(self, output_attention, use_gpu):
        self.model = model
        self.data = data
        self.features = features
        self.pred_len = pred_len
        self.label_len = label_len
        self.train_epochs = train_epochs
        self.use_amp = use_amp

        self.output_attention = output_attention
        self.use_gpu = use_gpu
        self.use_multi_gpu = use_multi_gpu
        self.devices = devices
        self.device_ids = device_ids
        self.clip = clip

