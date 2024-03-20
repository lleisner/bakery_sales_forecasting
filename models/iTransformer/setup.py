from models.iTransformer.data_loader import ITransformerData

data_loader = ITransformerData(
                data_path = 'data/tide_data.csv',
                datetime_col='datetime',
                numerical_cov_cols=['gaestezahlen', 'holidays', 'temperature', 'precipitation', 'cloud_cover', 'wind_speed'],
                categorical_cov_cols=None,
                cyclic_cov_cols=['wind_direction'],
                timeseries_cols=None,
                train_range=(0, 9015),
                val_range=(9016, 10947),
                test_range=(10948, 12879),
                hist_len=42,
                pred_len=16,
                stride=1,
                batch_size=32,
                epoch_len=None,
                val_len=None,
                normalize=True,
            )

dataset = data_loader.get_data()
print(dataset)
print(dataset.columns)

train, val, test = data_loader.get_train_test_splits()
print(train)

def inspect_dataset_shapes(dataset, num_batches_to_inspect=1):
    for i, batch in enumerate(dataset.take(num_batches_to_inspect)):
        batch_x, batch_y, batch_x_mark = batch
        print(f"Batch {i + 1}:")
        # If X_batch and y_batch are tensors or numpy arrays, you can directly access their shapes
        print("X and x mark shape:", batch_x.shape, batch_x_mark.shape, batch_y.shape)


def get_last_batch(dataset):
    last_batch = None
    number_of_batches = 0
    for batch in dataset:
        second_to_last = last_batch
        last_batch = batch
        number_of_batches += 1
    batch_x, batch_y, batch_x_mark = last_batch
    second_x, second_y, second_x_mark = second_to_last
    print("shape of the last X batch:", batch_x.shape, batch_y.shape, batch_x_mark.shape)
    print("shape of second to last batch", second_x.shape, second_y.shape, second_x_mark.shape)
    print("number of batches in X:", number_of_batches)

        
inspect_dataset_shapes(val, 3)

get_last_batch(val)
