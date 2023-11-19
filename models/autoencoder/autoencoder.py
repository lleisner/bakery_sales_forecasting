import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout


# Define the Encoder model using the Functional API
def build_encoder(input_shape, encoding_dim: int, layers: list, dropout: float=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for layer in layers:
        x = Dense(layer, activation='relu')(x)
        x = Dropout(dropout)(x)

    encoded = Dense(encoding_dim, activation='sigmoid')(x)
    encoder = Model(inputs, encoded, name='encoder')
    return encoder

# Define the Decoder model using the Functional API
def build_decoder(output_shape, encoding_dim: int, layers: list, dropout: float=0):
    inputs = Input(shape=(encoding_dim,))
    x = inputs
    for layer in layers:
        x = Dense(layer, activation='relu')(x)
        x = Dropout(dropout)(x)

    decoded = Dense(output_shape, activation='sigmoid')(x)
    decoder = Model(inputs, decoded, name='decoder')
    return decoder

# Define the AutoEncoder model using the encoder and decoder
def build_autoencoder(input_shape, encoding_dim: int=2, layers: list=[256, 128, 64]):
    inputs = Input(shape=input_shape)
    encoder = build_encoder(input_shape, encoding_dim, layers)
    decoder = build_decoder(input_shape, encoding_dim, list(reversed(layers)))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    
    autoencoder = Model(inputs, decoded, name='autoencoder')
    return autoencoder


