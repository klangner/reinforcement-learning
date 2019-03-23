from agents.dqn import QNetwork


def test_input_output_size():
    num_actions = 2
    network = QNetwork((4,), num_actions)
    y = network.predict([1,1,1,1])
    assert num_actions == len(y)