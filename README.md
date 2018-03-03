# awesome-machine-learning

Disclaimer: Very NN/DNN (vision) and RL (videogames/robotics) oriented, perhaps biased by DRL. Repo name may be subject to change.

# General

## General Frameworks
- [scikit-learn](http://scikit-learn.org) (API: Python)
- [scikit-image](http://scikit-image.org) (API: Python)

## General Books / Papers
- Jake VanderPlas, "Python Data Science Handbook", 2017. ([Safari](http://proquest.safaribooksonline.com/book/programming/python/9781491912126)) (API: Python)
- https://github.com/terryum/awesome-deep-learning-papers#new-papers

## General Other tools
- DLPaper2Code: Auto-generation of Code from Deep Learning Research Papers: https://arxiv.org/abs/1711.03543
- you can download the raw source of any arxiv paper. Click on the "Other formats" link, then click "Download source"
- http://www.arxiv-sanity.com/

# Neural Networks (NN) and Deep Neural Networks (DNN) 

## NN/DNN Frameworks
Here's a good overview [presentation](https://project.inria.fr/deeplearning/files/2016/05/DLFrameworks.pdf) ([permalink](https://github.com/jgvictores/awesome-machine-learning/blob/b16fadd3c56ce0d7fb3614cb63f155d5c2d4af81/doc/inria2007DLFrameworks.pdf)), and here's a very full [Docker](https://github.com/ufoym/deepo) with (tensorflow sonnet torch keras mxnet cntk chainer theano lasagne caffe caffe2). Attempling to order by current popularity:
- [Keras](https://keras.io) (layer over: TensorFlow, theano...) (API: Python)
   - https://en.wikipedia.org/wiki/Keras
   - Tutorials
      - https://www.datacamp.com/community/tutorials/deep-learning-python
      - https://elitedatascience.com/keras-tutorial-deep-learning-in-python
      - https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
- Torch (API: Lua)
   - https://github.com/junyanz/CycleGAN (but links to implementations with theano, etc)
   - https://github.com/karpathy/char-rnn
   - https://github.com/luanfujun/deep-photo-styletransfer
- [TensorFlow](https://www.tensorflow.org) (low-level)  (API: Python most stable)
   - https://github.com/llSourcell/YOLO_Object_Detection (but the original uses Darknet)
   - https://github.com/tkarras/progressive_growing_of_gans
   - https://github.com/mlberkeley/Creative-Adversarial-Networks
   - http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks
- [Chainer](http://www.chainer.org) ([GitHub](https://github.com/chainer/chainer)) (API: Python) (Define-by-Run rather than Define-and-Run)
- [Sonnet](https://deepmind.com/blog/open-sourcing-sonnet/) (by DeepMind) (layer over: TensorFlow)
   - https://github.com/deepmind/sonnet
- [MXNet](https://github.com/llSourcell/MXNet) (by Apache)
- [Darknet](https://pjreddie.com/darknet)
   - https://pjreddie.com/darknet/yolo
- [OpenNN](http://www.opennn.net)
   - Old work by Martin Stoelen
- [PyBrain](http://www.pybrain.org)
   - Old work by Santiago Morante
- theano (very used, but down here because [MILA stopped developing](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ))
   - Still many tutorials: https://github.com/lisa-lab/DeepLearningTutorials

## NN/DNN Architectures
- https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html
- http://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/

## NN/DNN Datasets
- http://www.image-net.org/
- https://en.wikipedia.org/wiki/AlexNet
- http://yann.lecun.com/exdb/mnist/

## NN/DNN Techniques Misc
- GANs.
- Transfer learning.
- Capsule Networks. R-CNN. (time sequences) RNN and more modern LSTM. (bad rumors) RBM.
- Dropout. Replay. 

# Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL)

## RL/DRL Frameworks
Attempling to order by current popularity:
- rllab ([GitHub](https://github.com/rll/rllab)) ([readthedocs](http://rllab.readthedocs.io)) (layer above: keras, and tensorflow in sandbox)
- Keras
   - https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec
   - https://github.com/haarnoja/sac
   - https://github.com/SoyGema/Startcraft_pysc2_minigames
- Torch
   - https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner
- ChainerRL ([GitHub](https://github.com/chainer/chainerrl)) (API: Python)

## RL/DRL Books
- Best ref is Andrew Ng thesis: www.cs.ubc.ca/~nando/550-2006/handouts/andrew-ng.pdf
- Reinforcement Learning: An Introduction: http://incompleteideas.net/book/bookdraft2017nov5.pdf (Richard S. Sutton is father of RL)

## RL/DRL Gyms
- https://gym.openai.com
- PySC2 ([GitHub](https://github.com/deepmind/pysc2)) (by DeepMind) (API: Python) (Blizzard StarCraft II Learning Environment (SC2LE) component)
- Atari-net ?

## RL/DRL Baselines
- https://github.com/openai/baselines

# Evolutionary Algorithms (EA)
Only accounting those with same objective as RL.
- https://blog.openai.com/evolution-strategies
- https://eng.uber.com/deep-neuroevolution/
- CMA 2006
