"""Contains arguments for command-line parsing."""
import argparse

def parse_args():
    """Parse arguments for `main.py`."""
    parser = argparse.ArgumentParser()

    # select model
    # parser.add_argument("--model", type=str, default=None,
                        # help="select one of the model types")

    # dataset
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="No. of points in the dataset (default 2000)")

    # topology
    parser.add_argument("--nodes", type=int, default=10,
                        help="No. of topology nodes (default 10)")
    parser.add_argument("--topo", type=str, default="fc",
                        help="Topology. One of: fc, random, ring (default fc)")

    # training setups
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default 0.01)")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size (default 20)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="No. of train epochs (default 5)")
    parser.add_argument("--local_steps", type=int, default=1,
                        help="No. of train steps per node per epoch (default 1)")
    parser.add_argument("--mixing_steps", type=int, default=1,
                        help="No. of params mixing steps per epoch (default 1)")

    #rng seend
    parser.add_argument("--seed", type=int, default=0)

    # how many times to repeat the train and test process
    # parser.add_argument("--nb_rounds", type=int, default=1)

    # save output in csv fomat
    # parser.add_argument("--csv", type=str, default=None)

    return parser.parse_args()
