import yaml
import logging as log
import torch.multiprocessing as mp
from trainer import Trainer
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)

with open('configs/params.yaml', 'r') as file:
    args = yaml.safe_load(file)
        
def main(rank):
    trainer = Trainer(rank, args)
    trainer.train()
    
if __name__ == "__main__":
    """Main program."""
    world_size = args["runconfig"]["world_size"]
    log.info(f'Parameters: \n{args}')
    mp.spawn(
        main,
        nprocs=world_size
    )

