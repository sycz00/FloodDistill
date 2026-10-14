

This package contains minimum code to run the flood segmentation models proposed in [Portalés et al. 2025](https://www.sciencedirect.com/science/article/pii/S003442572500286X#bib1). The main dependencies of this package are pytorch and the [georeader](https://github.com/spaceml-org/georeader) package which only depends on rasterio and geopandas libraries.

The notebook [run_in_gee_image.ipynb](https://github.com/kipoju/udl4fl/blob/main/notebooks/run_in_gee_image.ipynb) has an example of running the model on a Sentinel-2 image downloaded from the Google Earth Engine:

<img src="https://raw.githubusercontent.com/kipoju/udl4fl/main/example_flood.png">

More information about the datasets used to train the models can be found in the paper and in the respective project page of each dataset. 

## Models
With this package, the following models can be loaded: (with the function [`udl4fl.load_model_by_name()`](https://github.com/kipoju/udl4fl/blob/b02d832f6b946331ba97ae327c29684a42856342/udl4fl/model_setup.py#L133))

* **sm_unet_s2** Model trained on the Sentinel-2 L1C bands `["B02", "B03", "B04", "B08", "B11", "B12"]` from the S1S2Water and WorldFloods datasets. 
* **sm_unet_s1** Model trained on the Sentinel-1 GRD data  (`[VV, VH]` channels) from the S1S2Water and Kuro Siwo datasets. 
* **mm_unet_s1s2**  Model trained on the Sentinel-2 L1C bands `["B02", "B03", "B04", "B08", "B11", "B12"]` from the S1S2Water and WorldFloods datasets and Sentinel-1 GRD data  (`[VV, VH]` channels) from the S1S2Water and Kuro Siwo datasets.

## Citation

If you find this code useful please cite:

```
@article{PORTALESJULIA2025114882,
title = {Understanding flood detection models across Sentinel-1 and Sentinel-2 modalities and benchmark datasets},
journal = {Remote Sensing of Environment},
volume = {328},
pages = {114882},
year = {2025},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2025.114882},
url = {https://www.sciencedirect.com/science/article/pii/S003442572500286X},
author = {Enrique Portalés-Julià and Gonzalo Mateo-García and Luis Gómez-Chova},
keywords = {Flood detection, Deep learning, Multimodal fusion, Multispectral, SAR, Sentinel-1, Sentinel-2},
}

```

## Acknowledgments

This research has been funded by the DEEPCLOUD project (PID2019-109026RB-I00) funded by the Spanish Ministry of Science and Innovation (MCIN/AEI/10.13039/501100011033) and the European Union (NextGenerationEU).

<img width="300" title="DEEPCLOUD project (PID2019-109026RB-I00, University of Valencia) funded by MCIN/AEI/10.13039/501100011033." alt="DEEPCLOUD project (PID2019-109026RB-I00, University of Valencia) funded by MCIN/AEI/10.13039/501100011033." src="https://www.uv.es/chovago/logos/logoMICIN.jpg">

