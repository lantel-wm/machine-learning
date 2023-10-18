import argparse
from utils.trainer import Trainer

def parse_args() -> argparse.Namespace:
    """ parse command line arguments

    Returns:
        argparse.Namespace: command line arguments
    """
    parser = argparse.ArgumentParser(description="A simple command-line argument parser")

    # Add arguments to the parser
    parser.add_argument('--batch-size', type=int, default=256, help="Batch size")
    # parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--epochs' , type=int, default=100, help="Training epochs")
    parser.add_argument('--num-workers', type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument('--data', type=str, default='./data/SRCNN-GainMat-F16-sob-inf101.yaml', help='Path to dataset metadata file.')
    parser.add_argument('--device', type=str, default='parallel', help='Device to use for training. 0, 1, 2, 3, cpu or parallel (all GPUs)')
    parser.add_argument('--path', type=str, default='./experiments', help='Experiment save path.')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name.')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training.')
    parser.add_argument('--early-stop', type=int, default=10, help='Early stop patience.')

    args = parser.parse_args()

    return args  


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
    
