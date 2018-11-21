# Deep-COOC.pytorch
Implementation of [Deep-COOC](https://www.csie.ntu.edu.tw/~cyy/publications/papers/Shih2017DCF.pdf) pytorch version

![Deep-Cooc model](/examples/deepcooc_model.png)

## Requirements
 - python3
 - [Pytorch 0.4](https://github.com/pytorch/pytorch#from-source)
 - [torchvision](https://github.com/pytorch/vision)
 - numpy
 - [tensorboard](https://www.tensorflow.org/install/), tensorboard only would be enough

## TODO
 - [ ] Reproduce same results
 - [ ] Combine many modules
 - [ ] Try with other dataset

## Current status
 - Model can't reproduce the results that described in the paper.
 - I think there is a lot of details like weights freezing, learning rate, decays, but I can't follows all the stuff.
 - There is a lots of things you can do to improve. Do tuning!

## Performance
 - Sometime Deep-cooc model well perform than benchmarks, it depends on hyperparams.
 - They saids that they've got 73.3% accuracy for ResNet-152 but I've got 83.14% after carefully tuning the model.
 - For now, I've got 83.4% accuracy for Deep-cooc model with some trials.

## Usage
 - For training a deepcooc model, do ./deepcooc_CUB2011.sh
 - You can modify some hyperparam in deepcooc_CUB2011.sh or you can also add new params. To see more details, refer train.py
