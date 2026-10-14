import torch
import os
from udl4fl.utils import AttrDict
from udl4fl.data_configs import (CHANNELS_CONFIGURATIONS, SENTINEL2_NORMALIZATION,  BANDS_S2, DEFAULT_CHANNEL_CONFIG)

from udl4fl.worldfloods_model import WorldFloodsModel, load_weights
from typing import (Dict, Iterable, List, NamedTuple, Optional,
                    Tuple, Union)
from udl4fl.worldfloods_model import configure_architecture

MODELS_NAME= {
    "sm_unet_s1" : {
                    "model_file": "sm_unet_s1.pt",
                    "config_file": "sm_unet_s1.json",
             "bands": [0,1], 
             "type": "weights"},
    "sm_unet_s2" : {
                    "model_file": "sm_unet_s2.pt",
                    "config_file": "sm_unet_s2.json",
             "bands": ["B08", "B04", "B03", "B02"],
             "type": "weights"},
    "mm_unet_s1s2" : {
                    "model_file": "mm_unet_s1s2.pt",
             "bands": "all",
                "config_file": "mm_unet_s1s2.json",
             "type": "weights"}
}



# U-Net inputs must be divisible by 8
SUBSAMPLE_MODULE = {
    "unet": 8,
    "unet_dropout": 8
}


def get_model(model_params:AttrDict,
              experiment_name:Optional[str]=None, normalized_data:bool=True, device = None) ->  WorldFloodsModel:
    """
    Creates a model from a model_params Dict. (i.e. config.model_params).
    if model_params is in test or deploy model it will load the weights in the model_params.model_folder

    Args:
        model_params:
        experiment_name:
        normalized_data:

    Returns:
        pytorch lightning model

    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if model_params.get("test", False) or model_params.get("deploy", False):
        assert experiment_name is not None, f"Expermient name must be set on test or deploy mode"

        model = WorldFloodsModel(model_params, normalized_data=normalized_data)

        path_to_models = os.path.join(model_params.model_folder, experiment_name, "model.pt").replace("\\", "/")
        model.load_state_dict(load_weights(path_to_models))
        model.normalization = model_params.normalization
        print(f"Loaded model weights: {path_to_models}")
        model.to(device)
        model.eval()
        return model

    elif model_params.get("train", False):

        return WorldFloodsModel(model_params)

    else:
        raise Exception("No model type set in config e.g model_params.test == True")

           
def get_channel_configuration_bands(channel_configuration:str, collection_name:str="S2",
                                    as_string:bool=False) -> Union[List[int], List[str]]:
    """
    Returns 0-based list of channels of a given configuration name
    """
    if collection_name == "S2":
        channel_indexes = CHANNELS_CONFIGURATIONS[channel_configuration]
        if as_string:
            return [BANDS_S2[idx] for idx in channel_indexes]
        else:
            return channel_indexes
    else:
        raise NotImplementedError(f"Collection {collection_name} not implemented")

def channel_configuration_wrapper(inputs: Dict = DEFAULT_CHANNEL_CONFIG):
    bands_return = dict()
    for key in inputs.keys():
        if inputs[key]['used']:
            if key == 'S2':
                bands_return[key] = get_channel_configuration_bands(inputs[key]['channel_configuration'], collection_name=key)
            elif key == 'S1':
                bands_return[key] = [0,1]
            else:
                raise ValueError("Neither 'S2' nor 'S1' are used in the input dictionary.")

    return bands_return
    
def download_weights(weights_file:str):
    """Download weights from HuggingFace

    Args:
        weights_file (str): path to weights file

    Raises:
        ImportError: if gdown is not installed
    
    Example:
        >>> download_weights("weights/s2models/cloudsen12.ckpt", MODELS_CLOUDSEN12['all']['url'])
    """
    from huggingface_hub import hf_hub_download
    if not os.path.exists(weights_file):
        model_file = os.path.basename(weights_file)
        dirdownload = os.path.dirname(weights_file)
        os.makedirs(dirdownload, exist_ok=True)
        model_file_path = hf_hub_download(repo_id="isp-uv-es/udl4fl_models",
                                          filename=model_file,
                                          local_dir=dirdownload,
                                          local_dir_use_symlinks=False)
        if model_file_path != weights_file:
            raise ValueError(f"Error downloading {weights_file} file is downloaded in {model_file_path}")
        if not os.path.exists(weights_file):
            raise ValueError(f"Error downloading {weights_file}")    
    
    return weights_file


def load_model_by_name(name: str, weights_folder: str = "cloudsen12_models", 
                               device: torch.device = torch.device("cpu"), normalized_data: bool = False) -> WorldFloodsModel  :
    """
    Downloads model weights and config from HuggingFace using the name in MODELS_NAME, loads the model, and returns it in eval mode.
    Args:
        name (str): Model name as in MODELS_NAME
        weights_folder (str): Local folder to store weights/config
        device (torch.device): Device to load the model on
    Returns:
        torch.nn.Module: Model loaded in eval mode
    """
    import json
    from huggingface_hub import hf_hub_download

    if name not in MODELS_NAME:
        raise ValueError(f"Model name {name} not in {MODELS_NAME.keys()}")

    model_info = MODELS_NAME[name]
    weights_file = os.path.join(weights_folder, model_info['model_file'])
    config_file = os.path.join(weights_folder, model_info['config_file'])

    # Download weights and config if not present
    download_weights(weights_file)
    if not os.path.exists(config_file):
        os.makedirs(weights_folder, exist_ok=True)
        hf_hub_download(repo_id="isp-uv-es/udl4fl_models", filename=model_info['config_file'], local_dir=weights_folder, local_dir_use_symlinks=False)

    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    config = AttrDict(config).from_nested_dicts(config)


    # Build model architecture
    # model = configure_architecture(config.model_params.hyperparameters)
    
    model = WorldFloodsModel(config.model_params, normalized_data=normalized_data)

    state_dict = load_weights(weights_file)
    model.load_state_dict(state_dict)
    model.normalization = config.model_params.normalization
    model.to(device)
    model.eval()
    return model
    
    