import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def split_sequence(sequence, n_input: int, n_output: int) -> np.array:
    """Splits a times series sequence into input and output sequences

    Args:
        sequence: a time series to be split
        n_input: number of input steps
        n_output: number of output steps

    Returns:
        two numpy arrays

    """
    x, y = [], []
    for i, _ in enumerate(sequence):
        input_end = i + n_input
        output_end = input_end + n_output - 1
        # stop if we reach end of the sequence
        if output_end > len(sequence):
            break

        seq_x, seq_y = sequence[i:input_end], sequence[input_end - 1 : output_end]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def plot_validation(model, val_set, y_test, i):
    """

    Args:
        model:
        val_set:
        y_test:
        i:

    Returns:

    """
    forecasts = model.predict(val_set, verbose=0)

    X_pred = forecasts[i].flatten()
    X_pred_len = len(val_set[i])

    model_output = pd.DataFrame(
        {"date": np.arange(X_pred_len, X_pred_len + len(X_pred)) - 1, "output": X_pred}
    ).set_index("date", drop=True)

    model_input = pd.DataFrame(
        {"date": range(X_pred_len), "input": val_set[i].flatten()}
    ).set_index("date", drop=True)

    expected = pd.DataFrame(
        {
            "date": np.arange(X_pred_len, X_pred_len + len(X_pred)) - 1,
            "input": y_test[i].flatten(),
        }
    ).set_index("date", drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(model_output, label="predicted", c="r")
    ax.plot(model_input, label="input")
    ax.plot(expected, label="expected", c="gray", ls="--")
    plt.legend()
    plt.show()


def train_val_split(data, n_steps_in: int, n_steps_out: int, n_samples: int):
    """

    Args:
        data:
        n_steps_in:
        n_steps_out:
        n_samples:

    Returns:

    """
    n_samples = n_samples - 1
    split_idx = len(data) - (n_steps_in + n_steps_out - 1) - n_samples
    data_train = data[:split_idx]
    data_valid = data[split_idx:]

    x_train_set, y_train_set = split_sequence(
        data_train.values, n_steps_in, n_steps_out
    )
    x_valid_set, y_valid_set = split_sequence(
        data_valid.values, n_steps_in, n_steps_out
    )

    print(
        f"Created {x_train_set.shape[0]} training samples, and {x_valid_set.shape[0]} validation samples."
    )
    return x_train_set, y_train_set, x_valid_set, y_valid_set
