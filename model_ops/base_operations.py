import tensorflow as tf
from abc import ABC, abstractmethod
from models import iTransformer


class BaseOps(ABC):
    def __init__(self, args, data_pipeline):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer
        }
        self.device = self._acquire_device()
        self.model = self.build_model()
        self.data_pipeline = data_pipeline
        # self.args parameters = use_gpu, use_multi_gpu, devices, gpu, model, device_ids

    @abstractmethod
    def build_model(self):
        pass

    def _acquire_device(self):
        if self.args.use_gpu:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                gpu_id = self.args.gpu if not self.args.use_multi_gpu else self.args.devices
                tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
                print(f'Use GPU: {physical_devices[gpu_id]}')
                return physical_devices[gpu_id]
            else:
                print('No GPU devices available. Switching to CPU.')
                return tf.device('CPU')
        else:
            print('Use CPU')
            return tf.device('CPU')

    def _acquire_data(self):
        return DataPipeline()

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass