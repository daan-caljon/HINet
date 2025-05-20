# HINet: Heterogeneous Interference Network
This repository provides the code for the paper *"Estimating treatment effects in networks using domain adversarial learning"*.

The structure of the code is as follows:
```
HINet/
|_ data/
  |_ semi_synthetic/                   
    |_ BC/
    |_ Flickr/
|_ scripts/
  |_ create_figures.py                    
  |_ main_sweep.py            # Script to run a wandb sweep                   
|_ src/
  |_ data/
    |_ data_generator.py      # Code to generate synthetic and semi-synthetic data
    |_ datatools.py                 
  |_ methods/
    |_ Attention_layer.py
    |_ Causal_models.py       #Implementation of HINet + other methods
    |_ utils.py
  |_ utils/
      |_ metrics.py/          # CNEE and PEHNE implementations
      |_ utils.py/
  |_ training.py              # Trainer class
```

## Installation.
The ```requirements.txt``` provides the necessary packages.
All code was written for ```python 3.12.3```.

## Usage
Download the data for the BC and Flickr datasets from [Google Drive](https://drive.google.com/drive/folders/1CGOKpd7NU-brk9PpiO6nJcVYp3idi97E?usp=sharing). The original Flickr and BC data comes from [this repo](https://github.com/rguo12/network-deconfounder-wsdm20). We use the same data as [Jiang & Sun (2022)](https://github.com/songjiang0909/Causal-Inference-on-Networked-Data).
Put the data in the ```data/semi_synthetic/``` folder. For the homophilous and BA dataset, set the parameter ```homophily``` to ```True``` or ```False```, respectively. 
Now, the results from the paper can be reproduced by setting the ```DIR``` variable to your directory and selecting the right parameters in the ```wandb``` sweep configuration.

## Acknowledgements
Our code builds upon the code from [Jiang & Sun (2022)](https://github.com/songjiang0909/Causal-Inference-on-Networked-Data). 

Jiang, S. & Sun, Y. (2022). Estimating causal effects on networked observational data via representation learning. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, (pp. 852–861).
