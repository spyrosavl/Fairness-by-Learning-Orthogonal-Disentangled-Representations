# Fairness by Learning Orthogonal Disentangled Representations
This is a PyTorch implementation for the models introduced in [Fairness by Learning Orthogonal Disentangled Representations](https://arxiv.org/abs/2003.05707). This work was submitted in [ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020). 

This project is using the PyTorch template provided by [github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template).

# Prerequisites
  * [Conda Enviroment Manager](https://docs.conda.io/en/latest/)

# Getting Started
  Create `fact20` conda enviroment: 
  ````
    conda env create -f environment.yml
  ````
  Activate conda enviroment:
  ````
    conda activate fact20
  ````
  Download datasets:
  * [German Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
  * [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)
  * [Yale Extended Dataset (cropped version)](http://vision.ucsd.edu/content/extended-yale-face-database-b-b)
  * CIFAR-10 & CIFAR-100 will be downloaded automatically on the first run. 

  Extract datasets in `data` folder. The `data` folder should look like:
  ````
  data/
    adult/
      adult.data
      adult.test
    german/
      german.data
    yale/
      yaleB01
      .
      .
      .
      yaleB39
  ````
# Running the experiments
````
python train.py --config $CONFIG_FILE
````
The ``$CONFIG_FILE`` contains all the hyperparams for the experimens. The config files included are:
* ``config-adult.json``
* ``config-german.json``
* ``config-yale.json``
* ``config-cifar10.json``
* ``config-cifar100.json``

## Folder Structure
  ```
  root folder/
  │
  ├── train.py - main script to start training
  │
  ├── config-*.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

# Config parameters
Config files are in `.json` format:
```javascript
{
    "name": "German", // training session name
    "n_gpu": 1, // number of GPUs to use for training.

    "arch": {
        "type": "TabularModel", // name of model architecture to train
        "args": {
            "input_dim" : 61,
            "hidden_dim" : 64,
            "z_dim" : 2, // latent representation dims
            "target_classes" : 1, // number of target classes
            "sensitive_classes" : 2 // number of sensitive classes
        }
    },
    "data_loader": {
        "type": "GermanDataLoader", // selecting data loader
        "args":{
            "data_dir": "data/german", // dataset path
            "batch_size": 64,
            "shuffle": true, // shuffle training data before splitting
            "validation_split": 0.1, // size of validation dataset. float(portion) or int(number of samples)
            "num_workers": 1 // number of cpu processes to be used for data loading
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.001, // learning rate
            "weight_decay": 0.0005, // (optional) weight decay
            "amsgrad": true
        }
    },
    "loss": "loss", // loss function defined in model/loss.py
    "metrics": [
        "accuracy", "sens_accuracy" // validation metrics defined in model/metric.py
    ],
    "trainer": {
        "epochs": 100, // number of training epochs
        "save_dir": "saved/", // checkpoints are saved in save_dir/models/name
        "save_period": 1, // save checkpoints every save_freq epochs
        "verbosity": 2, // 0: quiet, 1: per epoch, 2: full

        "monitor": "min val_loss", // mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 0, // number of epochs to wait before early stop. set 0 to disable.

        "tensorboard": true,  // enable tensorboard visualization

        // lambda, gamma and step_size parameters as defined in the original paper
        "lambda_e" : 1.0,
        "lambda_od" : 0.01,
        "gamma_e" : 2.0,
        "gamma_od" : 1.4,
        "step_size" : 30
    }
}
```

# Authors
  * Spyros Avlonitis
  * Alexander Papadatos
  * Danai Xezonaki
