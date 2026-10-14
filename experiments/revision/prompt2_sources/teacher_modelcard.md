---
license: apache-2.0
library_name: terratorch
language:
- en
tags:
- Pytorch
- segmentation
- Flood mapping
- Sentinel-2
- Geospatial
- Foundation model
---
### Model and Inputs
The pretrained [Prithvi-EO-2.0-300M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL) model is finetuned to segment the extent of floods on Sentinel-2 images from the [Sen1Floods11 dataset](https://github.com/cloudtostreet/Sen1Floods11). 

The dataset consists of 446 labeled 512x512 chips that span all 14 biomes, 357 ecoregions, and 6 continents of the world across 11 flood events. The benchmark associated to Sen1Floods11 provides results for fully convolutional neural networks trained in various input/labeled data setups, considering Sentinel-1 and Sentinel-2 imagery. 

We use the following six bands for flood mapping: Blue, Green, Red, Narrow NIR, SWIR, SWIR 2.

Labels represent no water (class 0), water/flood (class 1), and no data/clouds (class -1).

The Prithvi-EO-2.0-300M-TL model was initially pretrained using a sequence length of 4 timestamps. Based on the characteristics of this benchmark dataset, we focus on single-timestamp segmentation. This demonstrates that our model can be utilized with an arbitrary number of timestamps during fine-tuning.

### Fine-tuning

The model was fine-tuned using [TerraTorch](https://github.com/IBM/terratorch):

```shell
terratorch fit -c sen1floods11.yaml
```

The configuration used for finetuning is available through this [config](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/sen1floods11.yaml).

### Inference and demo

A **demo** running this model is available **[here](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Sen1Floods11-demo)**.

This repo includes an inference script that allows running the flood model for inference on Sentinel-2 L1C images.

```shell
python inference.py --data_file examples/India_900498_S2Hand.tif
```

### Feedback

Your feedback is invaluable to us. If you have any feedback about the model, please feel free to share it with us. You can do this by submitting issues on GitHub or start a discussion on HuggingFace.

### Citation

If this model helped your research, please cite [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732) in your publications.

```
@article{Prithvi-EO-V2-preprint,    
    author          = {Szwarcman, Daniela and Roy, Sujit and Fraccaro, Paolo and Gíslason, Þorsteinn Elí and Blumenstiel, Benedikt and Ghosal, Rinki and de Oliveira, Pedro Henrique and de Sousa Almeida, João Lucas and Sedona, Rocco and Kang, Yanghui and Chakraborty, Srija and Wang, Sizhe and Kumar, Ankur and Truong, Myscon and Godwin, Denys and Lee, Hyunho and Hsu, Chia-Yu and Akbari Asanjan, Ata and Mujeci, Besart and Keenan, Trevor and Arévolo, Paulo and Li, Wenwen and Alemohammad, Hamed and Olofsson, Pontus and Hain, Christopher and Kennedy, Robert and Zadrozny, Bianca and Cavallaro, Gabriele and Watson, Campbell and Maskey, Manil and Ramachandran, Rahul and Bernabe Moreno, Juan},
    title           = {{Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications}},
    journal         = {arXiv preprint arXiv:2412.02732},
    year            = {2024}
}
```
