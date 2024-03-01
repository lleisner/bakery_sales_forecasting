import tensorflow as tf
import numpy as np
import os
import sys
import json

from tqdm import tqdm
from absl import logging

def training(dtl, model, lr, train_epochs, sum_dir):
    
    min_num_epochs = 5
    patience = 20
    num_split = 1
    
    
    step = tf.Variable(0)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=30 * dtl.train_range[1]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1e3)
    
    best_loss = np.inf
    pat = 0
    mean_loss_array = []
    iter_array = []
    os.makedirs(sum_dir, exist_ok=True)
    summary = Summary(sum_dir)
    
    while step.numpy() < train_epochs + 1:
        ep = step.numpy()
        logging.info('Epoch %s', ep)
        sys.stdout.flush()
        
        iterator = tqdm(dtl.tf_dataset(mode='train'), mininterval=2)
        for i, batch in enumerate(iterator):
            past_data = batch[:3]
            future_features = batch[4:6]
            tsidx = batch[-1]
            loss = model.train_step(
                past_data, future_features, batch[3], tsidx, optimizer
            )
            summary.update({'train/reg_loss': loss, 'train/loss': loss})
            if i % 100 == 0:
                mean_loss = summary.metric_dict['train/reg_loss'].result().numpy()
                mean_loss_array.append(mean_loss)
                iter_array.append(i)
                iterator.set_description(f'Loss {mean_loss:.4f}')
            step.assign_add(1)
            val_metrics, val_res, val_loss = model.eval(
                dtl, 'val', num_split=num_split
            )
            test_metrics, test_res, test_loss = model.eval(
                dtl, 'test', num_split=num_split
            )
            logging.info('Val Loss: %s', val_loss)
            logging.info('Test Loss: %s', test_loss)
            tracked_loss = val_metrics['rmse']
            if tracked_loss < best_loss and ep > min_num_epochs:
                best_loss = tracked_loss
                pat = 0
                
                with open(os.path.join(sum_dir, 'val_pred.npy'), 'wb') as fp:
                    np.save(fp, val_res[0][:, 0 : -1 : dtl.pred_len])
                with open(os.path.join(sum_dir, 'val_true.npy'), 'wb') as fp:
                    np.save(fp, val_res[1][:, 0 : -1 : dtl.pred_len])
                    
                with open(os.path.join(sum_dir, 'test_pred.npy'), 'wb') as fp:
                    np.save(fp, test_res[0][:, 0 : -1 : dtl.pred_len])
                with open(os.path.join(sum_dir, 'test_true.npy'), 'wb') as fp:
                    np.save(fp, test_res[1][:, 0 : -1 : dtl.pred_len])
                with open(os.path.join(sum_dir, 'test_metrics.json'), 'w') as fp:
                    json.dump(test_metrics, fp)
                logging.info('saved best result so far at %s', sum_dir)
                logging.info('Test metrics: %s', test_metrics)
            else:
                pat += 1
                if pat > patience:
                    logging.info('Early stopping')
                    break
            summary.write(step=step.numpy())
            
            
            
            
class Summary:
  """Summary statistics."""

  def __init__(self, log_dir):
    self.metric_dict = {}
    self.writer = tf.summary.create_file_writer(log_dir)

  def update(self, update_dict):
    for metric in update_dict:
      if metric not in self.metric_dict:
        self.metric_dict[metric] = tf.keras.metrics.Mean()
      self.metric_dict[metric].update_state(values=[update_dict[metric]])

  def write(self, step):
    with self.writer.as_default():
      for metric in self.metric_dict:
        tf.summary.scalar(metric, self.metric_dict[metric].result(), step=step)
    self.metric_dict = {}
    self.writer.flush()