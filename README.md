# ORGAN: Object-centric Representation Learning using Cycle-Consistent Generative Adversarial Networks

## Overview

ORGAN (Object-centric Representation Learning using Cycle-Consistent Generative Adversarial Networks) is a framework designed for learning object-centric representations through generative adversarial training. This project includes tools for dataset creation, model training, and evaluation.

## Setup

Install the required packages using Conda. Ensure you have Conda installed on your system. Execute the following commands to create and activate the environment:

```
conda create --name organ_env -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 python=3.10
conda activate organ_env
pip install -r requirements.txt
```

If you plan to use Weights and Biases (wandb) for logging and tracking experiments, install it separately:

```
pip install wandb
```

## Dataset Creation

Datasets can be generated using data.py. The script also generates a PDF file illustrating the dataset. To create a dataset, use the following command:

```
python3 data.py 
    --dataset       <Name of the dataset to be created> 
    --path          <Directory where the dataset will be stored>
```

### Available Datasets

The following datasets are implemented:

- sprites
- sprites_test
- sprites_big
- sprites_big_test
- sprites_test_very_big
- sprites_big_few_features
- mnist
- tetris
- cells
- cells_big
- cells_large
- cells_very_large

### Dataset Downloads

Some datasets require manual downloads:

1. Cells Dataset  
   - Source: https://www.kaggle.com/datasets/irfanbinazmi/blood-cell
   - Choose the images in /blood cell/blood/normal -imej x cantik
   - Path: Save the images in ./Data/Cells/Data.

2. Tetris dataset (officially Tetrominoes)  
   - Source: https://github.com/google-deepmind/multi_object_datasets 
   - Path: Save tfrecords in ./Data/

## Training

To train a model from scratch, use main.py. Before starting, configure the paths in config.py. Use the following command to train:

```
python3 main.py 
    --name          <Experiment Name, leave empty if not using wandb> 
    --config        <Index of the config to be loaded from config.py> 
    --gpu           <GPU ID to use> 
    --numWorkers    <Number of dataloader workers> 
    --seed          <Seed for model initialization> 
    --savePath      <Directory to save the trained model> 
    --wandb_path    <Directory to save wandb log files> 
    --deterministic <Set True for deterministic training, slower but reproducible>
```

### Available Configurations

Predefined configurations include:

- sprites
- mnist
- tetris
- cells

## Evaluation

For evaluation and visualization of model performance, use the provided scripts: eval.py, plot.py, and eval_feature_space.py.

### 1. Evaluation Script (eval.py)

```
python3 eval.py 
    --gpus          <GPUs to use> 
    --numWorkers    <Number of dataloader workers> 
    --k             <Number of top-k patches proposed> 
    --checkDis      <Set to evaluate the discriminator> 
    --path          <Model directory> 
    --batch_size    <Batch size> 
    --dataPath      <Path to the dataset to be evaluated>
```

### 2. Plotting Script (plot.py)

```
python3 plot.py 
    --path          <Model directory> 
    --dataPath      <Path to the dataset to be evaluated> 
    --no_ground     <Set if the list does not correspond to the ground truth>
```

## Example
To train a model on the sprites dataset, use the following command lines:

```
python3 data.py --dataset sprites --path ./Data/sprites
python3 data.py --dataset sprites_test --path ./Data/sprites_test
python3 main.py --config sprites --transformSpace
```
While plots and evaluation are done automatically, there is the option to this step manually:

```
python3 eval.py --path /path/to/your/model/directory --dataPath ./Data/sprites_test --k 10
python3 plot.py --path /path/to/your/model/directory --dataPath ./Data/sprites_test
```

## Notes and Best Practices

1. Config File Management: Ensure that paths in config.py are updated for your environment.
2. Reproducibility: Use the --seed and --deterministic options for reproducible results, though at the cost of slower performance.
3. Environment Dependencies: Always check compatibility of GPU and CUDA versions when setting up the environment.
