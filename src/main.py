import keras
import mlflow.keras
import matplotlib.pyplot as plt

from utils import plot_validation, train_val_split, prepare_data
from models import build_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


PAX = prepare_data(data_path="air-traffic-passenger-statistics.csv",
                   col1="Activity Period",
                   col2="Passenger Count")

n_steps_in = 12
n_steps_out = 12

X_train, y_train, X_valid, y_valid = train_val_split(PAX, n_steps_in, n_steps_out, 15)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

with mlflow.start_run():

    loss_metric = "mse"
    lr = 0.002
    epochs = 2_000
    patience = 1_000
    n_features = 1

    callback_1 = EarlyStopping(monitor="val_loss",
                               patience=patience)
    callback_2 = ReduceLROnPlateau(monitor="val_loss",
                                   factor=0.5,
                                   patience=300,
                                   verbose=0)

    model = build_model(n_steps_in, n_steps_out, n_features)

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
    mlflow.log_param("loss_func", loss_metric)

    # mlflow.keras.log_model(model, "keras-model")

plot_1 = plot_validation(model, X_valid, y_valid, 0)
plt.show()
plot_2 = plot_validation(model, X_valid, y_valid, 1)
plt.show()
