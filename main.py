import yaml
import logging as log
import torch.multiprocessing as mp
from trainer import Trainer

# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)


if __name__ == "__main__":
    """Main program."""
    with open('configs/params.yaml', 'r') as file:
        args = yaml.safe_load(file)
    log.info(f'Parameters: \n{args}')
    mp.spawn(
        Trainer,
        args=(*args),
        nprocs=4
    )

