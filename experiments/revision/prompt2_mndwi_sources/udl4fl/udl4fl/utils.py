from typing import Callable, List, Union
from tqdm import tqdm
import torch
import numpy as np
import itertools
import os
import fsspec
from pathlib import Path
from typing import Optional

REQUESTER_PAYS_DEFAULT = True
def get_filesystem(path: Union[str, Path], requester_pays:Optional[bool]=None):
    if requester_pays is None:
        requester_pays = REQUESTER_PAYS_DEFAULT
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0], requester_pays=requester_pays)
    else:
        # use local filesystem
        return fsspec.filesystem("file")

class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed like attributes
        (as well as normally).
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dicts(data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dicts(data[key])
                             for key in data})



@torch.no_grad()
def padded_predict(predfunction: Callable, module_shape: int) -> Callable:
    """
    This function is needed for U-Net like models that require the shape to be multiple of 8 (otherwise there is an
    error in concat layer between the tensors of the upsampling and downsampling paths).

    Args:
        predfunction:
        module_shape:

    Returns:
        Function that pads the input if it is not multiple of module_shape

    """
    def predict(x: torch.Tensor):
        """

        Args:
            x:

        Returns:

        """
        # actually this wont work for mmunet_md because it expects the keys to be S1 or S2
        if isinstance(x, torch.Tensor): 
            shape_tensor = np.array(list(x.shape))[2:].astype(np.int64)
            x = {0:x}
            
        elif isinstance(x, dict):
            k = 'S1' if 'S1' in x.keys() else 'S2'
            shape_tensor = np.array(list(x[k].shape))[2:].astype(np.int64)
            
        shape_new_tensor = np.ceil(shape_tensor.astype(np.float32) / module_shape).astype(np.int64) * module_shape

        if np.all(shape_tensor == shape_new_tensor):
            return predfunction(x)

        pad_to_add = shape_new_tensor - shape_tensor
        refl_pad_layer = torch.nn.ReflectionPad2d((0, pad_to_add[1], 0, pad_to_add[0]))

        refl_pad_result = {k:refl_pad_layer(x[k]) for k in x.keys() if k!='input_flag'}
        if 'input_flag' in x.keys():
            refl_pad_result['input_flag'] = x['input_flag']
            
        pred_padded = predfunction(refl_pad_result)
        slice_ = (slice(None),
                  slice(None),
                  slice(0, shape_new_tensor[0]-pad_to_add[0]),
                  slice(0, shape_new_tensor[1]-pad_to_add[1]))

        return pred_padded[slice_]

    return predict

@torch.no_grad()
def predbytiles(pred_function: Callable[[torch.Tensor], torch.Tensor], batch_dict: Union[torch.Tensor, List[torch.Tensor]],
                tile_size=1280, pad_size=32, device=torch.device("cpu"),
                disable_pbar:bool=False) -> torch.Tensor:
    """
    Apply a pred_function (usually a torch model) by tiling the input_batch array.
    The purpose is to run `pred_function(input_batch)` avoiding memory errors.
    It tiles and stiches the pateches with padding using the strategy of: https://arxiv.org/abs/1805.12219

    Args:
        pred_function: pred_function to call
        input_batch: torch.Tensor in BCHW format
        tile_size: Size of the tiles.
        pad_size: each tile is padded before calling the pred_function.
        device: Device to save the predictions

    Returns:
        torch.Tensor in BCHW format (same B, H and W as input_batch)

    """
    pred_continuous_tf = None
    input_batch = [batch_dict[k] for k in batch_dict.keys() if k!='input_flag']
    assert input_batch[0].dim() == 4, "Expected batch of images"

    indexes_list = list(itertools.product(range(0, input_batch[0].shape[0]),
                                     range(0, input_batch[0].shape[2], tile_size),
                                     range(0, input_batch[0].shape[3], tile_size)))
    
    for b, i, j in tqdm(indexes_list, desc="Predicting by tiles",
                        disable=disable_pbar or (len(indexes_list) == 1)):

        slice_current = (slice(i, min(i + tile_size, input_batch[0].shape[2])),
                         slice(j, min(j + tile_size, input_batch[0].shape[3])))
        slice_pad = (slice(max(i - pad_size, 0), min(i + tile_size + pad_size, input_batch[0].shape[2])),
                     slice(max(j - pad_size, 0), min(j + tile_size + pad_size, input_batch[0].shape[3])))

        slice_save_i = slice(slice_current[0].start - slice_pad[0].start,
                             None if (slice_current[0].stop - slice_pad[0].stop) == 0 else slice_current[0].stop -
                                                                                           slice_pad[0].stop)
        slice_save_j = slice(slice_current[1].start - slice_pad[1].start,
                             None if (slice_current[1].stop - slice_pad[1].stop) == 0 else slice_current[1].stop -
                                                                                           slice_pad[1].stop)

        slice_save = (slice_save_i, slice_save_j)

        slice_prepend = (slice(b, b + 1), slice(None))
        slice_current = slice_prepend + slice_current
        slice_pad = slice_prepend + slice_pad
        slice_save = slice_prepend + slice_save

        vals_to_predict = {k:batch_dict[k][slice_pad].to(device) for k in batch_dict.keys() if k!='input_flag'}
        if 'input_flag' in batch_dict.keys():
            vals_to_predict['input_flag'] = batch_dict['input_flag'].to(device)

        cnn_out = pred_function(vals_to_predict).cpu()

        assert cnn_out.dim() == 4, "Expected 4-band prediction (after softmax)"

        if pred_continuous_tf is None:
            pred_continuous_tf = torch.zeros((input_batch[0].shape[0], cnn_out.shape[1],
                                              input_batch[0].shape[2], input_batch[0].shape[3]),
                                             device=torch.device('cpu'))

        pred_continuous_tf[slice_current] = cnn_out[slice_save]

    return pred_continuous_tf