from keras.layers import LSTM, Dense, Dropout, LeakyReLU, RepeatVector, TimeDistributed
from keras.models import Sequential


def build_model(n_steps_in: int, n_steps_out: int, n_features: int):
    """build a sequential model

    Args:
        n_steps_in: number of input steps
        n_steps_out: number of output steps (forecast)
        n_features: number of features (uni-variate/multivariate)

    Returns: a model of a DNN with a given architecture

    """
    leak_ratio = 0.1
    drop_ratio = 0.2
    n_1 = 64
    n_2 = 32
    n_3 = 16

    model = Sequential()

    # encoder
    model.add(LSTM(n_1, input_shape=(n_steps_in, n_features), return_sequences=True))
    model.add(LeakyReLU(alpha=leak_ratio))
    model.add(Dropout(drop_ratio))

    model.add(LSTM(n_2, return_sequences=True))
    model.add(LeakyReLU(alpha=leak_ratio))
    model.add(Dropout(drop_ratio))

    model.add(LSTM(n_3))
    model.add(LeakyReLU(alpha=leak_ratio))
    model.add(Dropout(drop_ratio))

    model.add(RepeatVector(n_steps_out))

    # decoder
    model.add(LSTM(n_3, return_sequences=True))
    model.add(LeakyReLU(alpha=leak_ratio))
    model.add(Dropout(drop_ratio))

    model.add(LSTM(n_2, return_sequences=True))
    model.add(LeakyReLU(alpha=leak_ratio))
    model.add(Dropout(drop_ratio))

    model.add(LSTM(n_1, return_sequences=True))
    model.add(LeakyReLU(alpha=leak_ratio))
    model.add(Dropout(drop_ratio))

    model.add(TimeDistributed(Dense(1)))

    return model