from src.utils import split_sequence


def test_split_sequence():
    test_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    inputs, outputs = split_sequence(test_seq, 5, 3)
    assert len(inputs[0]) == 5
    assert len(outputs[0]) == 3
    assert inputs[0][0] == 1
    assert outputs[-1][-1] == 10
