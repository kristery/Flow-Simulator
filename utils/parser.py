import argparse

def parser():
    parser = argparse.ArgumentParser(description='Traffic Signal Control via Reinforcement Learning')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                        help='gae (default: 0.97)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                        help='damping (default: 1e-1)')
    parser.add_argument('--seed', type=int, default=8021, metavar='N',
                        help='random seed (default: 8021')
    parser.add_argument('--batch-size', type=int, default=4000, metavar='N',
                        help='size of a single batch')
    parser.add_argument('--num-epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train an expert')
    parser.add_argument('--hidden-dim', type=int, default=100, metavar='H',
                        help='dimension of hidden layers of architectures')
    parser.add_argument('--pg-type', type=str, default='ppo',
                        help='possible options: td3, trpo, ppo')
    parser.add_argument('--num-episodes', type=int, default=500)
    parser.add_argument('--td3-iter', type=int, default=100)
    parser.add_argument('--ps', type=str, default='')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    return args

