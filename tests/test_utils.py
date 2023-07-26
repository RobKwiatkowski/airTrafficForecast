from src.utils import _split_sequence, prepare_data
import pytest


def test_split_sequence():
    test_seq = [1, 2, 3, 4, 5]

    inputs, outputs = _split_sequence(test_seq, 3, 2)

    assert len(inputs[0]) == 3
    assert len(outputs[0]) == 2
    assert inputs[0][0] == 1
    assert outputs[-1][-1] == 5


def test_prepare_data_error():
    with pytest.raises(FileNotFoundError) as exc_info:
        exception_raised = prepare_data("not_exist_path", "A", "B")
        assert FileNotFoundError == exception_raised


def test_prepare_data_read():
    result = prepare_data("test_data/test_data_1.csv", "A", "B")
    assert result.index.name == "A"
    assert len(result.columns) == 1
    assert result.columns[0] == "B"


def test_prepare_data_scaling():
    result = prepare_data("test_data/test_data_1.csv", "A", "B")
    assert min(result["B"]) == 0
    assert max(result["B"]) == 1
