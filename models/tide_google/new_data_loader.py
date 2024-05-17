import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from models.data_loader import DataLoader
from models.tide_google.time_features import TimeCovariates

class TiDEData(DataLoader):
    def __init__(self, permute=False, *args, **kwargs):
        """
        Initialize objects

        Args:
            permute (bool, optional): Whether to permute the time series in training or not. Defaults to False.
            *args: Additional positional arguments for the parent class DataLoader.
            **kwargs: Additional keyword arguments for the partent class DataLoader.
        """
        
        super().__init__(*args, **kwargs)
        self.permute = permute
        self.batch_size = min(self.batch_size, len(self.ts_cols))
        
    def _add_missing_cols(self):
        if not self.num_cov_cols:
            self.data_df['ncol'] = np.zeros(self.data_df.shape[0])
            self.num_cov_cols = ['ncol']
        if not self.cat_cov_cols:
            self.data_df['ccol'] = np.zeros(self.data_df.shape[0])
            self.cat_cov_cols = ['ccol']
            
    def _create_temporal_features(self):
        index = self.data_df.index
        date_index = index.union(
        pd.date_range(
            index[-1] + pd.Timedelta(1, freq=self.freq),
            periods=self.pred_len + 1,
            freq=self.freq,
        )
        )
        self.time_df = TimeCovariates(date_index, normalized=self.normalize).get_covariates()
        
    def _create_lagged_features(self, include_mean_max_min=False):
        pass
        
    def _arrange_data(self):
        self.data_mat = self.data_df[self.ts_cols].to_numpy().transpose()
        self.data_mat = self.data_mat[:, 0 : self.test_range[1]]
        self.time_mat = self.time_df.to_numpy().transpose()
        self.num_feat_mat = self.data_df[self.num_cov_cols].to_numpy().transpose()
        self.cat_feat_mat, self.cat_sizes = self._get_cat_cols(self.cat_cov_cols)

    def _get_cat_cols(self, cat_cov_cols):
        """Get categorical columns."""
        cat_vars = []
        cat_sizes = []
        for col in cat_cov_cols:
            dct = {x: i for i, x in enumerate(self.data_df[col].unique())}
            cat_sizes.append(len(dct))
            mapped = self.data_df[col].map(lambda x: dct[x]).to_numpy().transpose()  # pylint: disable=cell-var-from-loop
            cat_vars.append(mapped)
        return np.vstack(cat_vars), cat_sizes

    def _normalize_data(self):
        self.scaler = StandardScaler()
        train_mat = self.data_mat[:, self.train_range[0] : self.train_range[1]]
        self.scaler = self.scaler.fit(train_mat.transpose())
        self.data_mat = self.scaler.transform(self.data_mat.transpose()).transpose()

    def _get_features_and_ts(self, dtimes, tsidx, hist_len=None):
        """Get features and ts in specified windows."""
        if hist_len is None:
            hist_len = self.hist_len
        data_times = dtimes[dtimes < self.data_mat.shape[1]]
        bdata = self.data_mat[:, data_times]
        bts = bdata[tsidx, :]
        bnf = self.num_feat_mat[:, data_times]
        bcf = self.cat_feat_mat[:, data_times]
        btf = self.time_mat[:, dtimes]
        if bnf.shape[1] < btf.shape[1]:
            rem_len = btf.shape[1] - bnf.shape[1]
            rem_rep = np.repeat(bnf[:, [-1]], repeats=rem_len)
            rem_rep_cat = np.repeat(bcf[:, [-1]], repeats=rem_len)
            bnf = np.hstack([bnf, rem_rep.reshape(bnf.shape[0], -1)])
            bcf = np.hstack([bcf, rem_rep_cat.reshape(bcf.shape[0], -1)])
        bfeats = np.vstack([btf, bnf])
        bts_train = bts[:, 0:hist_len]
        bts_pred = bts[:, hist_len:]
        bfeats_train = bfeats[:, 0:hist_len]
        bfeats_pred = bfeats[:, hist_len:]
        bcf_train = bcf[:, 0:hist_len]
        bcf_pred = bcf[:, hist_len:]
        return bts_train, bts_pred, bfeats_train, bfeats_pred, bcf_train, bcf_pred
    

    def train_gen(self):
        num_ts = len(self.ts_cols)
        perm = np.arange(
            self.train_range[0] + self.hist_len,
            self.train_range[1] - self.pred_len,
        )
        perm = np.random.permutation(perm)
        hist_len = self.hist_len
        logging.info('Hist len: %s', hist_len)
        if not self.steps_per_epoch:
            self.steps_per_epoch = len(perm)
        print("steps per epoch: ", self.steps_per_epoch)
        
        for idx in perm[0:self.steps_per_epoch]:
            #for i in range(num_ts // self.batch_size + 1):
            for i in range(num_ts//self.batch_size+1):
                if self.permute:
                    tsidx = np.random.choice(num_ts, size=self.batch_size, replace=False)
                else:
                    tsidx = np.arange(num_ts)
                dtimes = np.arange(idx - hist_len, idx + self.pred_len)
                (
                    bts_train,
                    bts_pred,
                    bfeats_train,
                    bfeats_pred,
                    bcf_train,
                    bcf_pred,
                ) = self._get_features_and_ts(dtimes, tsidx, hist_len)

                all_data = [
                    bts_train,
                    bfeats_train,
                    bcf_train,
                    bts_pred,
                    bfeats_pred,
                    bcf_pred,
                    tsidx,
                ]
                yield tuple(all_data)

    def test_val_gen(self, mode='val'):
        if mode == 'val':
            start = self.val_range[0]
            end = self.val_range[1] - self.pred_len + 1
        elif mode == 'test':
            start = self.test_range[0]
            end = self.test_range[1] - self.pred_len + 1
        else:
            raise NotImplementedError('Eval mode not implemented')
        num_ts = len(self.ts_cols)
        hist_len = self.hist_len
        logging.info('Hist len: %s', hist_len)
        perm = np.arange(start, end)
        if not self.validation_steps:
            self.validation_steps = len(perm)
        print("validation steps: ", self.validation_steps)
        
        for idx in perm[0:self.validation_steps]:
            for batch_idx in range(0, num_ts, self.batch_size):
                tsidx = np.arange(batch_idx, min(batch_idx + self.batch_size, num_ts))
                dtimes = np.arange(idx - hist_len, idx + self.pred_len)
                (
                    bts_train,
                    bts_pred,
                    bfeats_train,
                    bfeats_pred,
                    bcf_train,
                    bcf_pred,
                ) = self._get_features_and_ts(dtimes, tsidx, hist_len)
                all_data = [
                    bts_train,
                    bfeats_train,
                    bcf_train,
                    bts_pred,
                    bfeats_pred,
                    bcf_pred,
                    tsidx,
                ]
                yield tuple(all_data)


    def tf_dataset(self, mode='train'):
        if mode == 'train':
            gen_fn = self.train_gen
        else:
            gen_fn = lambda: self.test_val_gen(mode)
        output_types = tuple(
            [tf.float32] * 2 + [tf.int32] + [tf.float32] * 2 + [tf.int32] * 2
        )

        ts, t_feats, n_feats, c_feats = self.data_mat.shape[0], self.time_mat.shape[0], self.num_feat_mat.shape[0], self.cat_feat_mat.shape[0]
        output_shapes = ((ts, self.hist_len), 
                        (t_feats + n_feats, self.hist_len), 
                        (c_feats, self.hist_len), 
                        (ts, self.pred_len), 
                        (t_feats + n_feats, self.pred_len), 
                        (c_feats, self.pred_len), 
                        (ts,))
        dataset = tf.data.Dataset.from_generator(gen_fn, output_types=output_types, output_shapes=output_shapes)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    

    def get_train_test_splits(self):
        train = self.tf_dataset(mode='train')#.repeat(5)
        val = self.tf_dataset(mode='val')#.repeat(5)
        test = self.tf_dataset(mode='test')#.repeat(5)
        return train, val, test
    
    @staticmethod
    def prepare_batch(bts_train, bfeats_train, bcfeats_train, bts_pred, bfeats_pred, bcfeats_pred, tsidx):
        past_data = (bts_train, bfeats_train, bcfeats_train)
        future_features = (bfeats_pred, bcfeats_pred)
        y_true = bts_pred
        inputs = (past_data, future_features, tsidx)
        return inputs, y_true
 
    """
    

    def _create_datasets(self, start, end, mode='train'):
        num_ts = len(self.ts_cols)
        perm = np.arange(start + self.hist_len, end - self.pred_len)
        if mode == 'train':
            perm = np.random.permutation(perm)

        all_data = []
        
        def get_tsidx():
            if self.permute and mode == 'train':
                return np.random.choice(num_ts, size=self.batch_size, replace=False)
            else:
                return np.arange(num_ts)

        def append_data(idx, tsidx):
            dtimes = np.arange(idx - self.hist_len, idx + self.pred_len)
            (
                bts_train,
                bts_pred,
                bfeats_train,
                bfeats_pred,
                bcf_train,
                bcf_pred,
            ) = self._get_features_and_ts(dtimes, tsidx, self.hist_len)
            all_data.append([
                bts_train,
                bfeats_train,
                bcf_train,
                bts_pred,
                bfeats_pred,
                bcf_pred,
                tsidx,
            ])

        if mode == 'train':
            for idx in perm:
                for _ in range(num_ts // self.batch_size + 1):
                    tsidx = get_tsidx()
                    append_data(idx, tsidx)
        else:
            start = self.val_range[0] if mode == 'val' else self.test_range[0]
            end = (self.val_range[1] if mode == 'val' else self.test_range[1]) - self.pred_len + 1
            perm = np.arange(start, end)
            for idx in perm:
                for batch_idx in range(0, num_ts, self.batch_size):
                    tsidx = np.arange(batch_idx, min(batch_idx + self.batch_size, num_ts))
                    append_data(idx, tsidx)

        return all_data


    def get_train_test_splits(self):
        train_data = self._create_datasets(self.train_range[0], self.train_range[1], mode='train')
        val_data = self._create_datasets(self.val_range[0], self.val_range[1], mode='val')
        test_data = self._create_datasets(self.test_range[0], self.test_range[1], mode='test')
        
        train_dataset = self.tf_dataset(train_data)
        val_dataset = self.tf_dataset(val_data)
        test_dataset = self.tf_dataset(test_data)
        
        return train_dataset, val_dataset, test_dataset

    def tf_dataset(self, data):
        def map_fn(bts_train, bfeats_train, bcf_train, bts_pred, bfeats_pred, bcf_pred, tsidx):
            past_data = (bts_train, bfeats_train, bcf_train)
            future_features = (bfeats_pred, bcf_pred)
            return (past_data, future_features, tsidx), bts_pred
        data = tf.ragged.constant(data)

        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(lambda features: map_fn(*features))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
        
    """