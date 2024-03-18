from models.iTransformer.data_loader import ITransformerData

data_loader = ITransformerData(
                data_path = 'data/tide_data.csv',
                datetime_col='datetime',
                numerical_cov_cols=['gaestezahlen', 'holidays', 'temperature', 'precipitation', 'cloud_cover', 'wind_speed'],
                categorical_cov_cols=None,
                cyclic_cov_cols=['wind_direction'],
                timeseries_cols=None,
                train_range=None,
                val_range=None,
                test_range=None,
                hist_len=None,
                pred_len=None,
                stride=None,
                batch_size=32,
                epoch_len=None,
                val_len=None,
                normalize=True,
            )

dataset = data_loader.get_data()
print(dataset)
