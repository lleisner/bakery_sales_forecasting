from models.tide_google.tide_model import TiDE
from tensorflow import keras

def build_tide(hp, learning_rate, seq_len, pred_len, num_ts, cat_sizes=[]):
    hidden_size = hp.Choice('hidden_size', values=[2**i for i in range(7, 10, 1)])  # 128, 256, 512
    decoder_output_dim = hp.Choice('decoder_output_dim', values=[2**i for i in range(2, 6)])  # 4, 8, 16, 32
    temporal_decoder_hidden = hp.Choice('final_decoder_hidden', values=[2**i for i in range(5, 8)])  # 32, 64, 128

    n_blocks = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    activation = hp.Choice('activation', values=['relu', 'gelu'])
    
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    
    model = TiDE(
        seq_len=seq_len,
        pred_len=pred_len,
        num_ts=num_ts,
        cat_sizes=cat_sizes,
        hidden_size=hidden_size,
        decoder_output_dim=decoder_output_dim,
        final_decoder_hidden=temporal_decoder_hidden,
        time_encoder_dims=[64,4],
        num_layers=n_blocks,
        dropout=dropout,
        activation=activation,
        transform=True,
        cat_emb_size=4,
        layer_norm=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse',
        metrics=['mae'],
        weighted_metrics=[],
    )

    return model
