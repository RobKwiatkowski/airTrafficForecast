"""The main script training the DNN model and doing predictions
"""

import keras
import matplotlib.pyplot as plt
import mlflow.keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models import build_model
from utils import plot_validation, prepare_data, train_val_split

PAX = prepare_data(
    data_path="air-traffic-passenger-statistics.csv", col1="Activity Period", col2="Passenger Count"
)

N_STEP_IN = 12
N_STEPS_OUT = 12

X_train, y_train, X_valid, y_valid = train_val_split(data=PAX,
                                                     n_steps_in=N_STEP_IN,
                                                     n_steps_out=N_STEPS_OUT,
                                                     val_samples=15)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

with mlflow.start_run():
    LOSS_METRIC = "mse"
    LR = 0.002
    EPOCHS = 2_000
    PATIENCE = 1_000
    N_FEATURES = 1

    callback_1 = EarlyStopping(monitor="val_loss", patience=PATIENCE)
    callback_2 = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=300, verbose=0)

    model = build_model(N_STEP_IN, N_STEPS_OUT, N_FEATURES)

    # training
    opt = keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss=LOSS_METRIC)

    mlflow.tensorflow.autolog()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        verbose=1,
        batch_size=80,
        callbacks=[callback_1, callback_2],
        workers=4,
    )

    # model.save("my_model.keras")

    mlflow.log_param("patience", PATIENCE)
    mlflow.log_param("n_input", N_STEP_IN)
    mlflow.log_param("loss_func", LOSS_METRIC)

    # mlflow.keras.log_model(model, "keras-model")

plot_1 = plot_validation(model, X_valid, y_valid, 0)
plt.show()
plot_2 = plot_validation(model, X_valid, y_valid, 1)
plt.show()
