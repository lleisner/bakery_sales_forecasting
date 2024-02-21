import numpy as np
import tensorflow as tf
import pandas as pd

def predict_sample(sample, model, settings, num_targets):
    index = sample.index[-settings.future_steps]
    
    X = sample.iloc[:, :-num_targets]
    y = sample.iloc[-settings.future_steps:, -num_targets:]
    

    x_s , x_mark = X.iloc[:, :num_targets].values, X.iloc[:, num_targets:].values
    x_s, x_mark = np.expand_dims(x_s, axis=0), np.expand_dims(x_mark, axis=0)
    x_s, x_mark = tf.convert_to_tensor(x_s), tf.convert_to_tensor(x_mark)

    prediction = model.predict((x_s, x_mark), batch_size=1)
    prediction = pd.DataFrame(np.squeeze(prediction), index=index)
    
    return prediction, y
