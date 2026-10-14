import torch
import numpy as np
import pytorch_lightning as pl
from typing import List, Optional, Dict, Tuple, Union
from udl4fl.utils import AttrDict
from udl4fl.utils import get_filesystem

from udl4fl.utils import predbytiles, padded_predict
from udl4fl.data_configs import DEFAULT_CHANNEL_CONFIG
from udl4fl.models import S1S2DualEncoder_MD, smp_Unet

# U-Net inputs must be divisible by 8
SUBSAMPLE_MODULE = {
    "unet": 32,
}

def configure_architecture(h_params:AttrDict) -> torch.nn.Module:
    
    architecture = h_params.get('model_type', 'linear')
    dict_num_channels = h_params.get('num_channels', 3)
    channel_config = h_params.get('channel_configuration', DEFAULT_CHANNEL_CONFIG)
    
    if architecture == "mmunet":
        assert channel_config['S1']['used'] and channel_config['S2']['used'], "MMunet requires both S2 and S1 used"
        num_channels = dict_num_channels['S2']
    else:
        if channel_config['S2']['used']:
            num_channels = dict_num_channels['S2']
        elif channel_config['S1']['used']:
            num_channels = dict_num_channels['S1']

    num_classes = h_params.get('num_classes', 2)
        
    if architecture == "md_mmunet":

        encoder = h_params.get("encoder", "resnet18")
        attention_type = h_params.get("attention_type", None)
        in_channels = h_params.get("num_channels", None)
        
        model = S1S2DualEncoder_MD(in_channels_s1=2, in_channels_s2=in_channels['S2'], num_classes=num_classes,
                                attention_type=attention_type, encoder_name=encoder,
                                fusion_operation=h_params.get("fusion_operation", "mean"))
        
    elif architecture == 'unet':
                   
        encoder = h_params.get("encoder", "resnet18")
        attention_type = h_params.get("attention_type", None)
        in_channels = h_params.get("num_channels", None) 
        model = smp_Unet(encoder, num_channels, num_classes, attention_type) # TODO si eso mirar el depth

    else:
        raise Exception(f'No model implemented for model_type: {h_params.model_type}')

    return model


class WorldFloodsModel(pl.LightningModule):
    """
    Model to do multiclass classification.
    It expects ground truths y (B, H, W) tensors to be encoded as: {0: invalid, 1: clear, 2:water, 3: cloud}
    The preds (model.forward(x)) will produce a tensor with shape (B, 3, H, W)
    """
    def __init__(self, model_params: AttrDict, normalized_data:bool=True):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = configure_architecture(h_params_dict)
        self.model_type = h_params_dict.get('model_type', 'linear')
        self.max_tile_size = h_params_dict.get('max_tile_size', 1280)
        
        self.weight_per_class = torch.Tensor(h_params_dict.get('weight_per_class',
                                                               [1 for i in range(self.num_class)]),
                                             device=self.device)
        self.normalized_data = normalized_data

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])
        channel_config = h_params_dict.get('channel_configuration', ['S2'])
        self.inputs = [i for i in channel_config.keys() if channel_config[i]['used']]
        
        model_class = 'unet' if 'unet' in self.model_type else self.model_type
        self.module = SUBSAMPLE_MODULE.get(model_class, 1)
        
        self.normalization = None

    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, 1, W, H) encoded as {0: invalid, 1: land, 2: water}
        """
        pass
            
    def forward(self, x: Union[torch.Tensor, list]) -> torch.Tensor:
        """

        Args:
            x: (B, num_channels, H, W) input BATCH dictionary with the inputs

        Returns:
            (B, 3, H, W) prediction of the network
        """

        return self.network(x)
    
    def predict(self, x) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                batch = {self.inputs[0]: x}
            else:
                batch = x
            
            if not self.normalized_data and self.normalization is not None:
                    for inputs in self.inputs:
                        mean =  torch.Tensor(self.normalization['mean'][inputs]).to(batch[inputs].device)[..., None, None]
                        std =  torch.Tensor(self.normalization['std'][inputs]).to(batch[inputs].device)[..., None, None]
                        batch[inputs] = (batch[inputs] - mean) / std

            x = [batch[i] for i in self.inputs if i in batch.keys()] # we used to predict using list of inputs but now models handle bath like dictionaries
    
            # check if the inputs need padding for Unet like models
            if self.module > 1:
                pred_function = padded_predict(self.forward, self.module)
            else:
                pred_function = self.forward
                
            # handle predbytiles  
            if any((s > self.max_tile_size for s in x[0].shape[2:])):
                logits = predbytiles(pred_function, batch, self.max_tile_size, pad_size=0,device=self.device, disable_pbar=True)
            else:
                logits = pred_function(batch)
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            return torch.argmax(probs, dim=1).to(torch.int8), probs
    

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        pass

    def configure_optimizers(self):
        pass 

    def wb_mask(self, bg_img, pred_mask, true_mask):
        pass
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
        }


METRIC_MODE = {
    "val_dice_loss": "min",
    "val_iou water": "max",
    "val_bce_land_water": "min",
    "val_bce_loss": "min",
    "val_Acc_land_water": "max"
}


def load_weights(path_weights:str, map_location="cpu"):
    fs = get_filesystem(path_weights)
    if fs.exists(path_weights):
        with fs.open(path_weights, "rb") as fh:
            weights = torch.load(fh, map_location=map_location)
        return weights

    raise ValueError(f"Pretrained weights file: {path_weights} does not exists")



        






