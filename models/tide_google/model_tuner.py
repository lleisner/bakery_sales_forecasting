from models.tide_google.tide_model import TiDE
from tensorflow import keras

def build_tide(hp, learning_rate, seq_len, pred_len, num_ts, cat_sizes=[]):
    hidden_size = hp.Choice('hidden_size', values=[2**i for i in range(6, 10)])  # 64, 128, 256, 512
    decoder_output_dim = hp.Choice('decoder_output_dim', values=[2**i for i in range(3, 8)])  # 8, 16, 32, 64, 128
    final_decoder_hidden = hp.Choice('final_decoder_hidden', values=[2**i for i in range(2, 7)])  # 4, 8, 16, 32, 64
    time_encoder_dims = [hp.Choice('time_encoder_dim1', values=[2**i for i in range(4, 8)]),  # 16, 32, 64, 128
                         hp.Choice('time_encoder_dim2', values=[2**i for i in range(0, 4)])]  # 1, 2, 4, 8
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    activation = hp.Choice('activation', values=['relu', 'gelu'])
    transform = hp.Boolean('transform')
    layer_norm = hp.Boolean('layer_norm')
    
    cat_emb_size = hp.Int('cat_emb_size', min_value=4, max_value=4, step=1)

    model = TiDE(
        seq_len=seq_len,
        pred_len=pred_len,
        num_ts=num_ts,
        cat_sizes=cat_sizes,
        hidden_size=hidden_size,
        decoder_output_dim=decoder_output_dim,
        final_decoder_hidden=final_decoder_hidden,
        time_encoder_dims=time_encoder_dims,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        transform=transform,
        cat_emb_size=cat_emb_size,
        layer_norm=layer_norm
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse',
        metrics=['mae'],
        run_eagerly=True,
    )

    return model
