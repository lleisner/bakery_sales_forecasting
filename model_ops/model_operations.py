
import tensorflow as tf
from model_ops.base_operations import BaseOps
import time

class ModelOps(BaseOps):
    def __init__(self, args):
        super(ModelOps, self).__init__(args)
        self.args = args




    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.args.device_ids)
            with mirrored_strategy.scope():
                model = model.build()
        return model

    def _select_optimizer(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
        return optimizer

    def _select_loss_function(self):
        loss_function = tf.keras.losses.MeanSquaredError()
        return loss_function

    def _process_attention_output(self, model, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.output_attention:
            return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


    def vali(self, vali_data, vali_loader, loss_function):
        total_loss = []
        self.model.compile(optimizer=self.optimizer, loss=loss_function)
        self.model.evaluate(vali_data) 

        for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
            # decoder input
            dec_inp = tf.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)
        
            # encoder - decoder
            if self.args.use_amp:
                with tf.keras.mixed_precision.experimental.Policy('mixed_float16'):
                    outputs = self._process_attention_output(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self._process_attention_output(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            # detach and move to cpu computation
          #  pred = tf.identity(outputs).numpy()  # Convert to NumPy array
           # true = tf.identity(batch_y).numpy()  # Convert to NumPy array

          #  pred = tf.convert_to_tensor(pred)  # Convert NumPy array back to TensorFlow tensor
         #   true = tf.convert_to_tensor(true)  # Convert NumPy array back to TensorFlow tensor

            loss = loss_function(pred, true)
            total_loss.append(loss)

        # Calculate average loss
        total_loss = tf.reduce_mean(total_loss)
        return total_loss



    @tf.function
    def custom_train_step(self, i, batch_x, batch_y, batch_x_mark, batch_y_mark):
        iter_count += 1

        batch_x = tf.cast(batch_x, dtype=tf.float32)
        batch_y = tf.cast(batch_y, dtype=tf.float32)
        batch_x_mark = tf.cast(batch_x_mark, dtype=tf.float32)
        batch_y_mark = tf.cast(batch_y_mark, dtype=tf.float32)

        dec_inp = tf.zeros_like(batch_y[:, -self.args.pred_len:, :])
        dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)
        
        with tf.GradientTape() as tape:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # restrict loss calculation to pred_len
            f_dim = 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            loss = loss_function(outputs, batch_y)

        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.numpy()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        gradients = tape.gradient(loss, self.model.trainable_variables)
        if self.args.use_amp:
            gradients = scaler.get_scaled_gradients(gradients)
        gradients = [tf.clip_by_value(grad, -self.args.clip, self.args.clip) for grad in gradients]
        model_optim.apply_gradients(zip(gradients, self.model.trainable_variables))
       
        return loss
        
    
    def train(self, data, num_epochs, callbacks):
        train, vali, test = self.data_pipeline.generate_data()

        time_now = time.time()
        train_steps = len(train)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',  # Monitor loss
            patience=100,         # Number of epochs with no improvement to wait
            restore_best_weights=True  # Restore the best weights when stopped
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        callbacks = tf.keras.callbacks.CallbackList([early_stopping, tensorboard_callback])

        optimizer = self._select_optimizer()
        loss_function = self._select_loss_function()

    #    path = os.path.join(self.args.checkpoints, setting)
     #   if not os.path.exists(path):
      #      os.makedirs(path)


        if self.args.use_amp:
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            scaler = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            losses = []
            train_loss = []
            test_losses = []

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                loss = self.custom_train_step(i, batch_x, batch_y, batch_x_mark, batch_y_mark)
                train_loss.append(loss)

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # Perform early stopping check and other callbacks after each epoch
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={'val_loss': current_val_loss})
                if callback.model.stop_training:
                    print("Training stopped by callback: ", type(callback).__name__)
                    break
            callbacks.on_epoch_end(epoch, logs={'val_loss': vali_loss})
            
            # If early stopping, break out of training
            if callbacks[0].model.stop_training:
                break 

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


