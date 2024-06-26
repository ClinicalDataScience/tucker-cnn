# Post-Training Network Compression for 3D Medical Image Segmentation: Reducing Computational Efforts via Tucker Decomposition

Code for the paper *Post-Training Network Compression for 3D Medical Image Segmentation: Reducing Computational Efforts via Tucker Decomposition*.

> We address the computational barrier of deploying advanced deep learning segmentation models in clinical settings by studying the efficacy of network compression through tensor decomposition. We propose a post-training Tucker factorization that enables the decomposition of pre-existing models to reduce computational requirements without impeding segmentation accuracy. We applied Tucker decomposition to the convolutional kernels of the TotalSegmentator (TS) model, an nnU-Net model trained on a comprehensive dataset for automatic segmentation of 117 anatomical structures. Our approach reduced the floating-point operations (FLOPs) and memory required during inference, offering an adjustable trade-off between computational efficiency and segmentation quality. This study utilized the publicly available TS dataset, employing various downsampling factors to explore the relationship between model size, inference speed, and segmentation performance. The application of Tucker decomposition to the TS model substantially reduced the model parameters and FLOPs across various compression rates, with limited loss in segmentation accuracy. We removed up to 88% of the model's parameters with no significant performance changes in the majority of classes after fine-tuning. Practical benefits varied across different graphics processing unit (GPU) architectures, with more distinct speed-ups on less powerful hardware. Post-hoc network compression via Tucker decomposition presents a viable strategy for reducing the computational demand of medical image segmentation models without substantially sacrificing accuracy. This approach enables the broader adoption of advanced deep learning technologies in clinical practice, offering a way to navigate the constraints of hardware capabilities. 

## Introduction

Welcome to our repository!
Our paper is focused on analyzing the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
model.
However, our own package can be applied to any [nnU-net](https://github.com/MIC-DKFZ/nnUNet) derivative.
We validated the functionality of the `tuckercnn` module on `TotalSegmentator` version `2.0.5`
and `nnunetv2` on version `2.1`.

In a nutshell, our approach uses Tucker matrix decomposition to decompose one heavy-weight
convolution operation into three separate light-weight convolutions.
This is done post-training on already existing and released models!
A decomposed model can yield practical **speedups over 2x** by having over **90% less parameters**.
A schematic overview of the procedure is given below.
For further details check out our paper.

<p align="center">
<img src=assets/tucker_highlevel.png />
</p>

## How to Install

Required packages are listed in the `requirements.txt` file.
Install our package including the dependencies with `pip`:

```shell
pip install .
```

We use Python 3.11.
We are currently working on an actual PyPI release, so stay tuned.
Alternatively, checkout how you can [install packages directly from GitHub](https://adamj.eu/tech/2019/03/11/pip-install-from-a-git-repository/).


## How to Use

Our core-mechanic (ab)use the dynamics of Python and monkey-patch functions of the
`nnunet` package during runtime.
We encapsulate the patches within a context manager, which allows an easy to use
interface:

```python
from tuckercnn import TuckerContext

with TuckerContext:
    <your typical nnunet inference code>
```
Our package modifies the `nnunet` prediction utility and transforms the network directly.
In the realms of the `TotalSegmentator`, we can utilize the exposed Python API as following:
```python
from totalsegmentator.python_api import totalsegmentator
from tuckercnn import TuckerContext

with TuckerContext:
    totalsegmentator(input=<input path>, output=<output path>)
```
In the above examples we use the default configuration. You can inspect the parameters with::
```python
from tuckercnn.monkey.config import DEFAULT_CONFIG_DICT
print(DEFAULT_CONFIG_DICT)
```
The `TuckerContext` object takes a dictionary as an argument to overwrite the `DEFAULT_CONFIG_DICT`.
The configuration has following components:
```python
TUCKER_CONFIG = {
    'tucker_args': {
        'rank_mode': Method to determine the internal dimensions of the Tucker core 
            tensor. You have th choice between "relative" (original dim * rank_factor)
            or "fixed" (the internal rank equals the "rank_min" argument),
        'rank_factor': For mode "relative" the chosen scaling factor for the Tucker 
            core tensor dimensions. (Base recommendation = 0.3),
        'rank_min': If "rank_factor" is "relative", the "rank_min" amounts to the minimal,
        dimension the Tucker core tensor can have, for "fixed", this equals the chosen dimension,
        'decompose': False creates a Tucker model with random weights, if True, the
        original model's weight are decomposed and placed in the Tucker model',
        'verbose': Print extended information about the conversion process or not,
    },
    'apply_tucker': Flag to turn off the Tucker model replacement without having to remove
    the TuckerContext object,
    'inference_bs': Inference batchsize,
    'ckpt_path': Path to a checkpoint or desired location for checkpoint saving,
    'save_model': Save the new Tucker model,
    'load_model': Load a Tucker model from a checkpoint,
}
```
The above declared dictionary can then be passed into `TuckerContext`:
```python
with TuckerContext(TUCKER_CONFIG):
    <your typical nnunet inference code>
```
We collected the previously described information in an illustrative example using a
random spleen segmentation sample from the `TotalSegmentator` train dataset.
It is contained in `example/example.py`.
We recommend to execute it via the supplied `run.sh`, as this sets the required
environment variables correctly:
```shell
./run.sh example/example.py
```

## How to handle Datasets

The general data handling follows the nnUNet concepts. For that you have to create a raw, preprocessed and results folder and set the nnUNet environment variables according the description in the [nnUNet repository](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md). Datasets are created in the raw folder and are then preproccessed with the corresponding nnUNet preprocess scripts. 

### TotalSegmentator Datasets

Download the TotalSegmentator data from [zenodo](https://zenodo.org/records/10047292). Then create a dataset folder ie. Dataset095-Organ. Run the the script `convert_ds.py <source_path> <destination_path> <label_map>` to combine the masks and copy the files to the folder. For that you need to set the nnUNet environment variables. The label_map in our example case would be `class_map_part_organs`. After that you have to run the nnUNet [preprocess scripts](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md) ie. `nnUNetv2_plan_and_preprocess -d 095`. For the 3mm weights you will need to resample the data. This can be done similarly to our notebook file `notebooks/resample_lowres.ipynb`.


## Scripts

We provide a directory `scripts` with different utilities to further investigate the
behavior of Tucker-decomposition, to reproduce our experiments and to aid the handling
of `TotalSegmentator`.
We recommend to launch the scripts via the `run.sh` file like `./run.sh <path-to-py-file>`

- `00_setup_ts.py`: Downloads the 3mm model and the organ 1.5mm model, which is the
necessary setup for benchmarking GPU performance.
- `01_single_benchmark.py`: Executes a single benchmark for one Tucker configuration.
- `02_full_benchmark.py`: Executes a series of benchmark for a grid of parameters.
- `03_predict_files.py`: Perform prediction using a Tucker-decomposed model on a series of
files. The script is called like `python 03_predict_files.py <path-to-config-file> <path-to-file-1> ... <path-to-file-n>`.
An example for a config file is in `configs/predict_config.yml`.
- `03_predict_folder.sh`: The downside of the above python file is its limit to being single-threaded.
You can't really use multiprocessing to load all the separate files, as one of PyTorch's
requirements consists of it being executed in the main process. This shell script circumvents
this issue, as it will split the files of a directory into separate chunks and passes them
to detached python processes.
- `04_eval.py'`: The previous script does only compute the predictions. This script
checks the achieved Dice and NSD scores. The metrics are saved in a `.csv` file.
The script takes the path to a config file like `configs/predict_config.yml` as the first
argument.
- `05_finetune.py`: This script allows to fine-tune a Tucker-decomposed network. It takes a
config file as an argument. An example config file is `configs/finetune_baseline.yml`.
- `06_tucker_rec_error.py`: Compute different reconstruction errors of the Tucker approximation
over a grid of various core tensor dimensions.

Additionally, the directory `visualization` provides code for the reproduction of 
the figures and graphs used in our paper.
