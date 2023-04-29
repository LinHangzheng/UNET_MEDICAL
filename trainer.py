from datetime import datetime
import os
import logging as log
import torch
import torch.optim as optim
from model import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import PerfTimer, Validator, compute_acu, CombinedLoss
from input import IRDatasetProcessor
import torch.distributed as dist
from collections import OrderedDict
import wandb
from tqdm import tqdm
# from einops import rearrange

class Trainer(object):
    """
    Base class for the trainer.
    The default overall flow of things:
    init()
    |- set_dataset()
    |- set_network()
    |- set_optimizer()
    |- set_renderer()
    |- set_logger()
    train():
        for every epoch:
            pre_epoch()
            iterate()
                step()
            post_epoch()
            |- log_tb()
            |- render_tb()
            |- save_model()
            |- resample()
            validate()
    Each of these submodules can be overriden, or extended with super().
    """

    #######################
    # __init__
    #######################
    
    def __init__(self, rank, params):
        """Constructor.
        
        Args:
            args (Namespace): parameters
            args_str (str): string representation of all parameters
            model_name (str): model nametag
        """
        #torch.multiprocessing.set_start_method('spawn')
        # multiprocessing.set_start_method('spawn')

        self.params = params
        self.wandb = params["wandb"] 
        self.ip = params["ip"]
        self.data_dir = params["train_input"]["data_dir"]
        self.input_dim = params["train_input"]["input_dim"]
        self.num_classes = params["train_input"]["num_classes"]
        self.image_shape = params["train_input"]["image_shape"]
        self.batch_size = params["train_input"]["batch_size"]
        
        self.optimizer = params["optimizer"]["optimizer_type"]
        self.loss_lambda = params["optimizer"]["loss_lambda"]
        self.lr = params["optimizer"]["lr"]
        self.weight_decay_rate = float(params["optimizer"]["weight_decay_rate"])
        self.epsilon = float(params["optimizer"]["epsilon"])

        self.model_type = params['runconfig']['model_type']
        self.dataset_type = params['runconfig']['dataset_type']
        self.pretrained = params['runconfig']['pretrained']
        self.pretrained_from_DDP = params["runconfig"]["pretrained_from_DDP"]
        self.logs = params["runconfig"]["logs"]
        self.valid = params["runconfig"]["valid"]
        self.valid_only = params["runconfig"]["valid_only"]
        self.epochs = params["runconfig"]["epochs"]
        self.steps_per_epoch = params["runconfig"]["steps_per_epoch"]
        self.save_checkpoints_epoch = params["runconfig"]["save_checkpoints_epoch"]
        self.model_path = params["runconfig"]["model_path"]
        self.valid_every = params["runconfig"]["valid_every"]
        self.save_as_new = params["runconfig"]["save_as_new"]
        self.world_size = params["runconfig"]["world_size"]
        self.mode = params["runconfig"]["mode"]
        self.find_unused_parameters = params["runconfig"]["find_unused_parameters"]
        
        
        
        self.rank = rank
        # Set device to use
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(self.rank if self.use_cuda else 'cpu')
        self.device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {self.device_name} with CUDA v{torch.version.cuda}')

        self.latents = None
        
        # In-training variables
        self.train_data_loader = None
        self.val_data_loader = None
        self.dataset_size = None
        self.log_dict = {}

        # Initialize
        self.set_process()
        self.set_timer()
        self.set_wandb()
        self.set_dataset()
        self.timer.check('set_dataset')
        self.set_network()
        self.set_optimizer()
        self.timer.check('set_optimizer')
        self.set_criteria()
        self.timer.check('set_criteria')
        self.set_logger()
        self.timer.check('set_logger')
        self.set_validator()
        self.timer.check('set_validator')

        
    #######################
    # __init__ helper functions
    #######################
    def set_process(self):
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.ip
        dist.init_process_group("nccl", 
                                rank=self.rank, 
                                world_size=self.world_size)
        
    def set_timer(self):
        self.timer = PerfTimer(self.rank, activate=False)
        self.timer.reset()
        
    def set_wandb(self):
        if self.rank ==0:
            wandb.init(group=self.wandb, 
                       project="breast_new",
                       entity="hangzheng", 
                       mode=None if self.wandb else "disabled" ) #,mode="disabled"
            wandb.config.update = self.params
        
    def set_dataset(self):
        """
        Override this function if using a custom dataset.  
        By default, it provides 2 datasets: 
            AnalyticDataset
            MeshDataset
        
        The code uses the mesh dataset by default, unless --analytic is specified in CLI.
        """
        # self.train_dataset = IRDataset(self.params, self.mode)
        # sampler = DistributedSampler(self.train_dataset,
        #                             num_replicas=self.world_size, 
        #                             rank=self.rank, 
        #                             shuffle=False, 
        #                             drop_last=False)
        # log.info("Dataset Size: {}".format(len(self.train_dataset)))
        
        # self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch, 
        #                                     shuffle=False, pin_memory=True, num_workers=0,sampler=sampler)
        if self.valid_only:
            return
        
        self.DatasetProcessor = IRDatasetProcessor(self.params)
        self.train_data_loader = self.DatasetProcessor.create_dataloader(
                                    is_training=True,rank=self.rank)
        self.data_iterator = iter(self.train_data_loader)
        self.timer.check('create_dataloader')
        log.info(f"Loaded dataset with size: {len(self.train_data_loader.dataset)}")
            
    def set_network(self):
        """
        Override this function if using a custom network, that does not use the default args based
        initialization, or if you need a custom network initialization scheme.
        """
        params_model = self.params['model']
        if self.model_type == 'GeneratorUNet':
            self.net = GeneratorUNet(in_channels=self.input_dim, out_channels=self.num_classes)
        elif self.model_type == 'UNet':
            self.net = UNet(in_ch=self.input_dim, out_ch=self.num_classes)
        elif self.model_type == 'Segmenter':
            model_cfg = self.params['model'].copy()
            model_cfg.update(self.params['vit'][model_cfg['backbone']])
            model_cfg['image_size'] = self.image_shape
            model_cfg['channels'] = self.input_dim
            self.net = create_segmenter(model_cfg)
        elif self.model_type == 'UNETR_2D':
            if self.params['model']['patch_size'] == 16:
                self.net = UNETR_patch16(img_shape=self.image_shape, 
                                input_dim=self.input_dim, 
                                output_dim=self.num_classes, 
                                embed_dim=params_model['embed_dim'], 
                                patch_size=params_model['patch_size'], 
                                num_heads=params_model['num_heads'], 
                                dropout=params_model['dropout'],
                                mlp_hidden=params_model['mlp_hidden'],
                                num_layers=params_model['num_layers'],
                                ext_layers=params_model['ext_layers'])
            elif self.params['model']['patch_size'] == 4:
                self.net = UNETR_patch4(img_shape=self.image_shape, 
                                input_dim=self.input_dim, 
                                output_dim=self.num_classes, 
                                embed_dim=params_model['embed_dim'], 
                                patch_size=params_model['patch_size'], 
                                num_heads=params_model['num_heads'], 
                                dropout=params_model['dropout'],
                                mlp_hidden=params_model['mlp_hidden'],
                                num_layers=params_model['num_layers'],
                                ext_layers=params_model['ext_layers'])
            elif self.params['model']['patch_size'] == 32:
                self.net = UNETR_patch32(img_shape=self.image_shape, 
                                input_dim=self.input_dim, 
                                output_dim=self.num_classes, 
                                embed_dim=params_model['embed_dim'], 
                                patch_size=params_model['patch_size'], 
                                num_heads=params_model['num_heads'], 
                                dropout=params_model['dropout'],
                                mlp_hidden=params_model['mlp_hidden'],
                                num_layers=params_model['num_layers'],
                                ext_layers=params_model['ext_layers'])
        elif self.model_type == 'UNETR':
            self.net = UNETR(img_shape=self.image_shape, 
                                input_dim=self.input_dim, 
                                output_dim=self.num_classes, 
                                embed_dim=params_model['embed_dim'], 
                                patch_size=params_model['patch_size'], 
                                num_heads=params_model['num_heads'], 
                                dropout=params_model['dropout'],
                                mlp_hidden=params_model['mlp_hidden'],
                                num_layers=params_model['num_layers'],
                                ext_layers=params_model['ext_layers'])
        elif self.model_type == 'SwinUNet':
            self.net = SwinUnet(img_size=self.image_shape[0],
                                num_classes=self.num_classes,
                                patch_size=params_model['patch_size'],
                                in_chans=self.input_dim,
                                embed_dim=params_model['embed_dim'],
                                window_size=params_model['window_size'],
                                )
        elif self.model_type == 'AttUNet':
            self.net = AttU_Net(img_ch=self.input_dim, output_ch=self.num_classes)
        elif self.model_type == 'INTRANET':
            self.net = INTRANET(img_shape=self.image_shape, 
                            input_dim=self.input_dim, 
                            output_dim=self.num_classes, 
                            embed_dim=params_model['embed_dim'], 
                            patch_size=params_model['patch_size'], 
                            num_heads=params_model['num_heads'], 
                            dropout=params_model['dropout'],
                            mlp_hidden=params_model['mlp_hidden'],
                            num_layers=params_model['num_layers'],
                            ext_layers=params_model['ext_layers'])
        if self.pretrained:
            state_dict = torch.load(self.pretrained)
            if not self.pretrained_from_DDP:
                self.net.load_state_dict(state_dict)
            else:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                    new_state_dict[name] = v
                self.net.load_state_dict(new_state_dict)
            log.info("Pretrained model loaded")

        self.net.to(self.rank)
        self.net = DDP(self.net, device_ids=[self.rank], output_device=self.rank,find_unused_parameters=self.find_unused_parameters)
        log.info("Total number of parameters: {}".format(sum(p.numel() for p in self.net.parameters())))

    def set_optimizer(self):
        """
        Override this function to use custom optimizers. (Or, just add things to this switch)
        """

        # Set geometry optimizer
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay_rate)
        elif self.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.8)
        elif self.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=self.epsilon, weight_decay=self.weight_decay_rate)
        else:
            raise ValueError('Invalid optimizer.')

    def set_criteria(self):
        weight_dice = 1/(1+self.loss_lambda)
        weight_ce = 1-weight_dice
        self.loss = CombinedLoss(weight_dice=weight_dice,
                                 weight_ce=weight_ce)
        
    def set_logger(self):
        """
        Override this function to use custom loggers.
        """
        self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.log_dir = os.path.join(self.logs, self.log_fname)
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Parameters', str(self.params))

        log.info('Model configured and ready to go')


    def set_validator(self):
        """
        Override this function to use custom validators.
        """
        if self.valid and self.rank ==0:
            self.validator = Validator(self.params, self.rank, self.net)
            log.info(f"Loaded validator with size: {len(self.validator.val_data_loader.dataset)}")       
        dist.barrier()     
    #######################
    # pre_epoch
    #######################

    def pre_epoch(self, epoch):
        """
        Override this function to change the pre-epoch preprocessing.
        This function runs once before the epoch.
        """
        
        self.net.train()
        
        # Initialize the dict for logging
        self.log_dict['cross_entropy_loss'] = 0
        self.log_dict['total_loss'] = 0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['training_dice'] = 0

        self.timer.check('pre_epoch done')

    #######################
    # iterate
    #######################b
    
    def iterate(self, epoch):
        """
        Override this if there is a need to override the dataset iteration.
        """
        # if we are using DistributedSampler, we have to tell it which epoch this is
        if self.world_size > 1:
            self.train_data_loader.sampler.set_epoch(epoch)
        for _ in range(self.steps_per_epoch):
            data = next(self.data_iterator)
            """
            Override this function to change the per-iteration behaviour.
            """
            # Map to device
            self.timer.check('inner_iter start')
            images = data[0].to(self.rank)
            labels = data[1].to(self.rank)
            self.timer.check('send to device')
            # Prepare for inference
            batch_size = images.shape[0]
            self.optimizer.zero_grad()
            self.timer.check('optimizer reset')
            # Calculate loss
            preds = self.net(images)
            del images
            self.timer.check('training')
            loss, dice, ce = self.loss(preds,labels)
            del preds, labels
            self.timer.check('get loss')
            # Update logs
            self.log_dict['cross_entropy_loss'] += ce.item()
            self.log_dict['total_loss'] += loss.item()*batch_size
            self.log_dict['training_dice'] += dice*batch_size
            self.log_dict['total_iter_count'] += batch_size
            
            loss /= batch_size
            self.timer.check('log update')
            # Backpropagate
            loss.mean().backward()
            self.optimizer.step()
            self.timer.check('inner done')
    #######################
    # post_epoch
    #######################
    
    def post_epoch(self, epoch):
        """
        Override this function to change the post-epoch post processing.
        By default, this function logs to Tensorboard, renders images to Tensorboard, saves the model,
        and resamples the dataset.
        To keep default behaviour but also augment with other features, do 
          
          super().post_epoch(self, epoch)
        in the derived method.
        """
        self.net.eval()
        
        if self.rank == 0:
            self.log_tb(epoch)
            if (epoch+1) % self.save_checkpoints_epoch == 0:
                self.save_model(epoch)
            
            self.timer.check('post_epoch done')
    
    #######################
    # post_epoch helper functions
    #######################

    def log_tb(self, epoch):
        """
        Override this function to change loss logging.
        """
        # Average over iterations

        log_text = 'EPOCH {}/{}'.format(epoch+1, self.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        self.log_dict['training_dice'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        log_text += ' | training dice {}'.format(self.log_dict['training_dice'].mean())
        log_text += ' | cross_entropy_loss {}'.format(self.log_dict['cross_entropy_loss'])
        wandb.log({"total loss": self.log_dict['total_loss'],
                   "training dice": self.log_dict['training_dice'].mean(),
                   "cross entropy_loss": self.log_dict['cross_entropy_loss']})
        log.info(log_text)

        # Log losses
        self.writer.add_scalar('Loss/total_loss', self.log_dict['total_loss'], epoch)

                
    def save_model(self, epoch):
        """
        Override this function to change model saving.
        """
        log_comps = self.log_fname.split('/')
        if len(log_comps) > 1:
            _path = os.path.join(self.model_path, *log_comps[:-1])
            if not os.path.exists(_path):
                os.makedirs(_path)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if self.save_as_new:
            model_fname = os.path.join(self.model_path, f'{self.log_fname}-{epoch}.pth')
        else:
            model_fname = os.path.join(self.model_path, f'{self.log_fname}.pth')
        
        log.info(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.net.state_dict(), model_fname)

        if self.latents is not None:
            model_fname = os.path.join(self.model_path, f'{self.log_fname}_latents.pth')
            torch.save(self.latents.state_dict(), model_fname)
        
    #######################
    # train
    #######################
    
    def train(self):
        """
        Override this if some very specific training procedure is needed.
        """

        if self.valid and self.valid_only:
            if self.rank ==0:
                self.validate(0)
            torch.distributed.barrier() 
            return

        for epoch in range(self.epochs):    
            self.timer.check('new epoch...')
            
            self.pre_epoch(epoch)

            if self.train_data_loader is not None:
                self.dataset_size = len(self.train_data_loader)
            
            self.timer.check('iteration start')

            self.iterate(epoch)

            self.timer.check('iterations done')

            self.post_epoch(epoch)

            if self.valid and (epoch+1) % self.valid_every == 0:
                if self.rank ==0:
                    self.validate(epoch)
                    self.timer.check('validate')    
        torch.distributed.barrier()
        self.cleanup()
        self.writer.close()
    
    #######################
    # validate
    #######################

    def validate(self, epoch):
        val_dict = self.validator.validate(epoch)
        log_text = 'EPOCH {}/{}'.format(epoch+1, self.epochs)
        wandb_dict = {}
        for k, v in val_dict.items():
            self.writer.add_scalar(f'Validation/{k}', v, epoch)
            log_text += ' | {}: {:.6f}'.format(k, v)
            wandb_dict[k] = v
        wandb.log(wandb_dict)
        log.info(log_text)
    
    def cleanup(self):
        dist.destroy_process_group()
