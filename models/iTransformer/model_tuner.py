from models.iTransformer.i_transformer import ITransformer
from tensorflow import keras


def build_itransformer(hp, learning_rate, seq_len, pred_len, num_ts):
    d_model = hp.Choice('d_model', values=[2**i for i in range(3, 10, 2)])  # 8, 32, 128, 512
    n_heads = hp.Choice('n_heads', values=[2**i for i in range(1, 4)])  # 2, 4, 8
    d_ff = hp.Choice('d_ff', values=[2**i for i in range(4, 11, 2)])  # 16, 64, 256, 1024
    e_layers = hp.Choice('e_layers', values=[i for i in range(1, 4)]) # 1, 2, 3
    
    dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    
    activation = hp.Choice('activation', values=['relu', 'gelu'])
    #clip = hp.Float('clip', min_value=0.0, max_value=1.0, step=0.1)
    clip = None
    use_norm = hp.Boolean('use_norm')

    model = ITransformer(
        seq_len=seq_len,
        pred_len=pred_len,
        num_ts=num_ts,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        e_layers=e_layers,
        dropout=dropout,
        output_attention=False,
        activation=activation,
        clip=clip,
        use_norm=use_norm
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse',
        metrics=['mae'],
        weighted_metrics=[],
    )

    return model
