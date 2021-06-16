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
    parser.add_argument("--iid", type=str, default="true",
                        help="Whether data is iid (default True)")

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
    parser.add_argument("--iters", type=int, default=5,
                        help="No. of train iterations (default 5)")
    parser.add_argument("--local_steps", type=int, default=1,
                        help="No. of train steps per node per epoch (default 1)")
    parser.add_argument("--mixing_steps", type=int, default=1,
                        help="No. of params mixing steps per epoch (default 1)")

    # rng seend
    parser.add_argument("--seed", type=int, default=0)

    # save output in csv fomat
    parser.add_argument("--csv", type=str, default=None,
                        help="Save to CSV output (default None)")

    return parser.parse_args()
