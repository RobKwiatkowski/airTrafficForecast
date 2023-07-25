from src.utils import _split_sequence


def test_split_sequence():
    test_seq = [1, 2, 3, 4, 5]

    inputs, outputs = _split_sequence(test_seq, 3, 2)

    assert len(inputs[0]) == 3
    assert len(outputs[0]) == 2
    assert inputs[0][0] == 1
    assert outputs[-1][-1] == 5
