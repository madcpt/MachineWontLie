# MachineWontLie

Project One for SJTU CS 385 Machine Learning, supervised by [Prof. Quanshi Zhang](http://qszhang.com/).

## Reproduction

I use YAML to save configurations. Experiments can be run as:

```bash
python3 train.py [train|debug] [CONFIGRATION]
```

There are two modes available: `train` and `debug`. The only difference between the two modes is that `debug` mode would not dump log files.

All configuration files can be found in `./config/`.

For example, to run experiments with Logistic Regression under training mode, use commands like:

```shell
python3 train.py train ./config/train_logistic_regression.yaml
```

## Tensorboard for Monitoring

Tensorboard serving with:

```shell
tensorboard --logdir ./runs
```

## Report

Report will be made public soon.