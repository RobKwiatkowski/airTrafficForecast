import keras
import mlflow.keras
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, LeakyReLU, RepeatVector, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from utils import plot_validation, train_val_split

PAX = pd.read_csv("air-traffic-passenger-statistics.csv")
PAX.loc[:, "Activity Period"] = pd.to_datetime(
    PAX.loc[:, "Activity Period"].astype(str), format="%Y%m"
)

df = PAX[["Activity Period", "Passenger Count"]]
df = df.groupby("Activity Period").sum()
df.sort_values(by="Activity Period", inplace=True)

scaler = MinMaxScaler()
scaler.fit(df)
df["Passenger Count"] = scaler.transform(df)

X_raw = df.copy()

n_steps_in = 12
n_steps_out = 12

X_train, y_train, X_valid, y_valid = train_val_split(X_raw, n_steps_in, n_steps_out, 15)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

n_features = 1

with mlflow.start_run():
    n_1 = 64
    n_2 = 32
    n_3 = 16
    loss_metric = "mse"
    lr = 0.002
    epochs = 4_000
    patience = 1_000
    leak_ratio = 0.1
    drop_ratio = 0.2

    callback_1 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
    callback_2 = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=300, verbose=0
    )

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

    # training
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss_metric)

    mlflow.tensorflow.autolog()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        verbose=1,
        batch_size=80,
        callbacks=[callback_1, callback_2],
        workers=4,
    )

    # model.save("my_model.keras")

    mlflow.log_param("patience", patience)
    mlflow.log_param("n_input", n_steps_in)
    mlflow.log_param("n_1", n_1)
    mlflow.log_param("n_2", n_2)
    mlflow.log_param("loss_func", loss_metric)

    # mlflow.keras.log_model(model, "keras-model")

plot_validation(model, X_valid, y_valid, 0)
plot_validation(model, X_valid, y_valid, 1)
