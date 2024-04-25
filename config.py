import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=5e-3)

    args = parser.parse_args()
    return args