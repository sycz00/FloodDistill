import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead

## SM MODEL

class smp_Unet(smp.UnetPlusPlus):
    def __init__(self, encoder, num_channels, num_classes, attention_type):
        super().__init__(
            encoder_name=encoder,  
            encoder_weights=None,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32,16),
            in_channels=num_channels,
            classes=num_classes,
            decoder_attention_type = attention_type,
            activation=None
        )
        self.keys_ignore = ['input_flag', 'mask']
    def forward(self, x):
        x_predict = [x[k] for k in x.keys() if not k in self.keys_ignore]
        outputs = super(smp_Unet, self).forward(*x_predict)
        return outputs
        
        
def configure_smp_model(model_name, encoder, num_channels, num_classes, depth = 5, activation = None, 
                        attention_type = None):
    
    if model_name == 'smp_unet':
        return smp_Unet(encoder, num_channels, num_classes, attention_type)
    else:
        raise ValueError(f'Model {model_name} not implemented. Choose from smp_unet or smp_deeplabv3')

### MULTIMODAL 

def concat_channels(x1, x2): 
    return [torch.cat((x1[i], x2[i]), dim=1) for i in range(len(x1))]

def mean_channels(x1, x2):
    fusion = lambda x1, x2: (x1 + x2) / 2
    return [torch.cat((x1[0], x2[0]), dim=1)] + [fusion(x1[i+1],x2[i+1]) for i in range(len(x1)-1)]

class S1S2DualEncoder_MD(torch.nn.Module):
    def __init__(self, in_channels_s1, in_channels_s2, num_classes, encoder_name= "efficientnet-b0",
                 fusion_operation = 'mean', attention_type = None):
        super(S1S2DualEncoder_MD, self).__init__()

        # Encoder 1
        self.encoder1 = smp.encoders.get_encoder(encoder_name, 
            in_channels=in_channels_s1,
            depth = 4,
            weights=None,
        )
            
        self.encoder2 = smp.encoders.get_encoder(encoder_name, 
            in_channels=in_channels_s2,
            depth = 4,
            weights=None,
        )
               
       ## DEFINE FUSION OPERATION
       
        self.fusion_operation = fusion_operation
        if self.fusion_operation == 'mean':
            self.fusion = mean_channels
            encoder_out_channels = self.encoder1.out_channels

        
        self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
            encoder_channels = encoder_out_channels,
            decoder_channels=(256, 128, 64, 32),
            n_blocks=4,
            use_batchnorm = False,
            attention_type = attention_type,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=32,
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, batch):
        
        
        if torch.all(batch['input_flag'][:,0] == 0):
            x = self.encoder2(batch['S2'])
        elif torch.all(batch['input_flag'][:,1] == 0):
            x = self.encoder1(batch['S1'])
        else:
            
            mask1 = batch['input_flag'][:,0].reshape(-1,1,1,1)
            mask2 = batch['input_flag'][:,1].reshape(-1,1,1,1)
            x1 = self.encoder1(batch['S1'])
            x2 = self.encoder2(batch['S2'])
            # perform a masked mean 
            
            fusion_func = lambda x1, x2, mask1, mask2: (mask1*x1 + mask2*x2) / (mask1 + mask2)
            x = [torch.cat((x1[0], x2[0]), dim=1)] + [fusion_func(x1[i+1],x2[i+1], mask1, mask2) for i in range(len(x1)-1)]

        # Decoder
        out = self.decoder(*x)

        out = self.segmentation_head(out)

        return out