import numpy as np

from agents import ReplayBuffer, build_training_set


def test_input_output_size():
    n_actions = 2
    qvalues = np.zeros((5, n_actions))
    qvalues2 = np.ones((5, n_actions))
    actions = np.array([0, 1, 0, 1, 0])
    rewards = np.array([1, 2, 3, 4, 5])
    dones = np.array([False, False, False, False, True])
    expected_y = np.array([[2, 0], [0, 3], [4, 0], [0, 5], [5, 0]])
    y = build_training_set(qvalues, qvalues2, actions, rewards, dones, 1.0)
    assert np.array_equal(y, expected_y), 'Wrong expected qvalue calculated'