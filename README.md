# awesome-deep-reinforcement-learning

Curated list for Deep Reinforcement Learning (DRL): frameworks, architectures, datasets, gyms, baselines...

To accomplish this, includes general Machine Learning (ML), Neural Networks (NN) and Deep Neural Networks (DNN) with many vision examples, and Reinforcement Learning (RL) with videogames/robotics examples. Some alternative Evolutionary Algorithms (EA) with similar objectives included too.

- [General Machine Learning (ML)](#general-machine-learning-ml)
   - [General ML Books / Papers](#general-ml-frameworks)
- [Neural Networks (NN) and Deep Neural Networks (DNN)](#neural-networks-nn-and-deep-neural-networks-dnn)
   - [NN/DNN Frameworks](#nndnn-frameworks)
   - [NN/DNN Architectures](#nndnn-architectures)
   - [NN/DNN Datasets](#nndnn-datasets)
   - [NN/DNN Techniques Misc](#nndnn-techniques-misc)
- [Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL)](#reinforcement-learning-rl-and-deep-reinforcement-learning-drl)
   - [RL/DRL Frameworks](#rldrl-frameworks)
   - [RL/DRL Books](#rldrl-books)
   - [RL/DRL Gyms](#rldrl-gyms)
   - [RL/DRL Baselines](#rldrl-baselines)
   - [RL/DRL Techniques Misc](#rldrl-techniques-misc)
- [Evolutionary Algorithms (EA)](#evolutionary-algorithms-ea)
- [Misc tools](#misc-tools)

# General Machine Learning (ML)

## General ML Frameworks
- [scikit-learn](http://scikit-learn.org) (API: Python)
- [scikit-image](http://scikit-image.org) (API: Python)

## General ML Books / Papers
- Jake VanderPlas, "Python Data Science Handbook", 2017. ([Safari](http://proquest.safaribooksonline.com/book/programming/python/9781491912126)) (API: Python)
- https://github.com/terryum/awesome-deep-learning-papers#new-papers

# Neural Networks (NN) and Deep Neural Networks (DNN) 

## NN/DNN Frameworks
Here's a good overview [presentation](https://project.inria.fr/deeplearning/files/2016/05/DLFrameworks.pdf) ([permalink](https://github.com/jgvictores/awesome-deep-reinforcement-learning/blob/143a885cc10b4331b9b3fa3e1a9436d5325676af/doc/inria2017DLFrameworks.pdf)), and here's a very full [Docker](https://github.com/ufoym/deepo) with (tensorflow sonnet torch keras mxnet cntk chainer theano lasagne caffe caffe2). Attempling to order by current popularity:
- [Keras](https://keras.io) (layer over: TensorFlow, theano...) (API: Python)
   - https://en.wikipedia.org/wiki/Keras
   - Examples/tutorials
      - https://github.com/keras-team/keras/blob/master/examples
      - https://www.datacamp.com/community/tutorials/deep-learning-python
      - https://elitedatascience.com/keras-tutorial-deep-learning-in-python (includes deepdream, but original uses Caffe)
- Torch (API: Lua)
   - https://github.com/junyanz/CycleGAN (but links to implementations with theano, etc)
   - https://github.com/karpathy/char-rnn
   - https://github.com/luanfujun/deep-photo-styletransfer
- [TensorFlow](https://www.tensorflow.org) (low-level)  (API: Python most stable)
   - https://github.com/llSourcell/YOLO_Object_Detection (but the original uses Darknet)
   - https://github.com/tkarras/progressive_growing_of_gans
   - https://github.com/mlberkeley/Creative-Adversarial-Networks
   - http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks
   - https://github.com/ckmarkoh/neuralart_tensorflow (looks like Deepdream, but short, and nice)
   - Another NN wrapper: Tensorpack ([GitHub](https://github.com/ppwwyyxx/tensorpack))
- [Caffe](http://caffe.berkeleyvision.org/) (API: Python, Matlab)
   - https://github.com/google/deepdream
   - FCIS, but released [MXNet version](https://github.com/msracver/FCIS), and [chainer re-implementation](https://github.com/knorth55/chainer-fcis) can also be found.
- [Chainer](http://www.chainer.org) ([GitHub](https://github.com/chainer/chainer)) (API: Python) (Define-by-Run rather than Define-and-Run) (in addition to chainerrl below, there is also a [chainercv](https://github.com/chainer/chainercv))
- [Sonnet](https://deepmind.com/blog/open-sourcing-sonnet/) ([GitHub](https://github.com/deepmind/sonnet)) (by DeepMind) (layer over: TensorFlow)
- [MXNet](https://github.com/llSourcell/MXNet) (by Apache)
- [Darknet](https://pjreddie.com/darknet)
   - https://pjreddie.com/darknet/yolo
- [OpenNN](http://www.opennn.net)
- [PyBrain](http://www.pybrain.org)
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
- Capsule Networks. R-CNN. (time sequences) RNN and more modern LSTM. RBM. Echo-state networks.
- GANs.
- Transfer learning.
- Max-pooling.
- Activation functions: ReLu.
- Optimization: ADAM.
- Dropout.

# Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL)

## RL/DRL Frameworks
Attempling to order by current popularity:
- rllab ([GitHub](https://github.com/rll/rllab)) ([readthedocs](http://rllab.readthedocs.io)) (officialy uses theano; in practice has some keras, tensorflow, torch, chainer...)
- Keras
   - https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec
   - https://github.com/haarnoja/sac
   - https://github.com/SoyGema/Startcraft_pysc2_minigames
- Torch
   - https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner
   - https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752 (originally in tensorflow)
- ChainerRL ([GitHub](https://github.com/chainer/chainerrl)) (API: Python)

## RL/DRL Books
- Reinforcement Learning: An Introduction: http://incompleteideas.net/book/bookdraft2017nov5.pdf (Richard S. Sutton is father of RL)
- Andrew Ng thesis: www.cs.ubc.ca/~nando/550-2006/handouts/andrew-ng.pdf

## RL/DRL Gyms
- [OpenAI Gym](https://gym.openai.com) ([GitHub](https://github.com/openai/gym)) ([docs](https://gym.openai.com/docs/))
   - https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym
   - Most recent [robotics environments](https://gym.openai.com/envs/#robotics) use MuJoCo, but old examples have been re-implemented in [roboschool](https://blog.openai.com/roboschool/) which is visually the same but MIT License.
- PySC2 ([GitHub](https://github.com/deepmind/pysc2)) (by DeepMind) (API: Python) (Blizzard StarCraft II Learning Environment (SC2LE) component)
- Atari-net ?

## RL/DRL Baselines
- https://github.com/openai/baselines

## RL/DRL Techniques Misc
- Batch: REINFORCE, Deep Q-Network (DQN), Truncated Natural Policy Gradient (TNPG), Trust Region Policy Optimization (TRPO), Reward-Weighted Regression, Relative Entropy Policy Search (REPS), Cross Entropy Method (CEM), Advantage-Actor-Critic (A2C), Asynchronous Advantage Actor-Critic (A3C), Actor-critic with Experience Replay (ACER), Actor Critic using Kronecker-Factored Trust Region (ACKTR), Generative Adversarial Imitation Learning (GAIL), HER, PPO1, PPO2.
- Online: Deep Determisitc Policy Gradient (DDPG).
- Experience Replay. 

# Evolutionary Algorithms (EA)
Only accounting those with same objective as RL.
- https://blog.openai.com/evolution-strategies
- https://eng.uber.com/deep-neuroevolution/
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

# Misc Tools
- DLPaper2Code: Auto-generation of Code from Deep Learning Research Papers: https://arxiv.org/abs/1711.03543
- Tip: you can download the raw source of any arxiv paper. Click on the "Other formats" link, then click "Download source"
- http://www.arxiv-sanity.com/
