# awesome-deep-reinforcement-learning

Curated list for Deep Reinforcement Learning (DRL): frameworks, architectures, datasets, gyms, baselines...

To accomplish this, includes general Machine Learning (ML), Neural Networks (NN) and Deep Neural Networks (DNN) with many vision examples, and Reinforcement Learning (RL) with videogames/robotics examples. Some alternative Evolutionary Algorithms (EA) with similar objectives included too.

- [General Machine Learning (ML)](#general-machine-learning-ml)
   - [General ML Books / Papers](#general-ml-frameworks)
- [Neural Networks (NN) and Deep Neural Networks (DNN)](#neural-networks-nn-and-deep-neural-networks-dnn)
   - [NN/DNN Software Frameworks](#nndnn-software-frameworks)
   - [NN/DNN Architectures](#nndnn-architectures)
   - [NN/DNN Datasets](#nndnn-datasets)
   - [NN/DNN Benchmarks](#nndnn-benchmarks)
   - [NN/DNN Pretrained](#nndnn-pretrained)
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

## NN/DNN Software Frameworks
Here's a good overview [presentation](https://project.inria.fr/deeplearning/files/2016/05/DLFrameworks.pdf) ([permalink](https://github.com/jgvictores/awesome-deep-reinforcement-learning/blob/143a885cc10b4331b9b3fa3e1a9436d5325676af/doc/inria2017DLFrameworks.pdf)), and here's a very full [Docker](https://github.com/ufoym/deepo) with (tensorflow sonnet torch keras mxnet cntk chainer theano lasagne caffe caffe2). Attempling to order by current popularity:
- [Keras](https://keras.io) (layer over: TensorFlow, theano...) (API: Python)
   - https://en.wikipedia.org/wiki/Keras
   - Examples/tutorials
      - https://github.com/keras-team/keras/blob/master/examples
      - https://www.datacamp.com/community/tutorials/deep-learning-python
      - https://elitedatascience.com/keras-tutorial-deep-learning-in-python (includes deepdream, but original uses Caffe)
- [PyTorch](https://pytorch.org/) (API: Python)
   - Used internally by http://www.fast.ai/
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
- [OpenCV](https://www.opencv.org) now has some DNN: https://docs.opencv.org/3.3.0/d2/d58/tutorial_table_of_content_dnn.html
- [OpenNN](http://www.opennn.net)
- [PyBrain](http://www.pybrain.org)
- theano (very used, but down here because [MILA stopped developing](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ))
   - Still many tutorials: https://github.com/lisa-lab/DeepLearningTutorials

## NN/DNN Architectures
- MobileNets (2017). Andrew Howard et Al; Google; "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"; [arxiv](https://arxiv.org/abs/1704.04861).
- YOLO9000 (2016). Joseph Redmond et Al; "YOLO9000: Better, Faster, Stronger"; [arxiv](https://arxiv.org/abs/1612.08242).
- SSD (2015). Wei Liu et Al; UNC, Zoox, Google, et Al; "SSD: Single Shot MultiBox Detector"; [arxiv](https://arxiv.org/abs/1512.02325). (orig: [caffe](https://github.com/intel/caffe/wiki/SSD:-Single-Shot-MultiBox-Detector))
- ResNet (2015). Kaiming He et Al; Microsoft Research; "Deep Residual Learning for Image Recognition"; [arxiv](https://arxiv.org/abs/1512.03385). Variants: ResNet50, ResNet101, ResNet152...
- GoogleNet (2014). Christian Szegedy et Al; Google, UNC; "Going Deeper with Convolutions"; [arxiv](https://arxiv.org/abs/1409.4842).
- VGGNet (2014). Karen Simonyan et Al; Visual Geometry Group (Oxford); "Very Deep Convolutional Networks for Large-Scale Image Recognition"; [arXiv](https://arxiv.org/abs/1409.1556). Input: 224 224 3. Variants: VGG16, VGG19.
- Network In Network (NIN) (2013). [arxiv](http://arxiv.org/abs/1312.4400).
- OverFeat (2013).
- AlexNet (2012). Alex Krizhevsky et Al; Toronto; "ImageNet Classification with Deep Convolutional Neural Networks"; [doi](https://doi.org/10.1145/3065386). ([wikipedia](https://en.wikipedia.org/wiki/AlexNet))
- LeNet (1998)
- Links to overviews:
   - https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html
   - http://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/
   - https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
- Visualization:
   - Keras: [tutorial](https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/) / [official](https://keras.io/visualization/)
   - Caffe: [Netscope](http://ethereon.github.io/netscope) / [cnnvisualizer](https://github.com/metalbubble/cnnvisualizer)

## NN/DNN Datasets
- [MNIST](http://yann.lecun.com/exdb/mnist/): Handwritten digits, set of 70000 examples, is a subset of a larger set available from NIST.
- [ImageNet](http://www.image-net.org/): Project organized according to the WordNet hierarchy (22000 categories). Includes SIFT features, bounding boxes, attributes. Currently over 14 million images, 21841 cognitive synonyms (synsets) indexed, goal of +1000 images per synset.
   - ImageNet Large Visual Recognition Challenge (ILSVRC): Goal of 1000 categories using +100000 test images. E.g. LS-LOC
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (Visual Object Classes)
- [COCO](http://cocodataset.org) (Common Objects in Context): 2014, 2015, 2017. Includes classes and annotations.
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60000 32x32 colour images (selected from MIT TinyImages) in 10 classes, with 6000 images per class
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html): 60000 32x32 colour images (selected from MIT TinyImages) in 100 classes containing 600 images per class, grouped into 20 superclasses
- [MIT Places](http://places.csail.mit.edu/)
- [MIT Moments](http://moments.csail.mit.edu/)
- Link to overview: https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research

## NN/DNN Benchmarks
- http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

## NN/DNN Pretrained
- CIFAR-10 and CIFAR-100:
   - CNN trained on CIFAR-100 tutorial: [keras](https://andrewkruger.github.io/projects/2017-08-05-keras-convolutional-neural-network-for-cifar-100)
   - VGG16 trained on CIFAR-10 and CIFAR-100: [keras](https://github.com/geifmany/cifar-vgg) / [keras CIFAR-10 weights](https://drive.google.com/open?id=0B4odNGNGJ56qVW9JdkthbzBsX28) / [keras CIFAR-100 weights](https://drive.google.com/open?id=0B4odNGNGJ56qTEdnT1RjTU44Zms)
- ImageNet and ILSVRC:
   - VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, Xception trained on ImageNet: [keras by keras](https://github.com/keras-team/keras/tree/master/keras/applications) ([permalink](https://github.com/keras-team/keras/tree/e15533e6c725dca8c37a861aacb13ef149789433/keras/applications)) / [keras by kaggle](https://www.kaggle.com/keras)
   - VGG16 trained on ImageNet (tutorial): [keras](https://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/)
   - VGGNet, ResNet, Inception, and Xception trained on ImageNet (tutorial): [keras](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)
   - VGG16 trained on ILSVRC: [caffe by original VGG author](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) / ported (tutorials): [tensorflow](https://www.cs.toronto.edu/~frossard/post/vgg16/) / [keras](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) / [keras ImageNet weights](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc)
- Misc:
   - Small examples: [keras by keras](https://github.com/keras-team/keras/tree/master/examples) ([permalink](https://github.com/keras-team/keras/tree/e15533e6c725dca8c37a861aacb13ef149789433/examples))
   - Model zoo: [caffe](https://github.com/BVLC/caffe/wiki/Model-Zoo)

## NN/DNN Techniques Misc
- Layers: Dense (Fully Connected), Convolutional, Pooling (Max, Average...), Normalisation. Note: in Keras, activation functions, dropout, etc are also implemented as layers.
- Activation functions: Rectified Linear Unit (ReLU), softmax, Exponential Linear Unit (ELU), scaled exponential linear unit (selu), softplus, softsign, tanh, sigmoid, hard sigmoid, linear, logit, Leaky version of a Rectified Linear Unit (LeakyReLU or LReLU), Parametrized or Parametric Rectified Linear Unit (PReLU), Swish. [wikipedia](https://en.wikipedia.org/wiki/Activation_function), [keras](https://keras.io/activations/), [keras (advanced)](https://keras.io/layers/advanced-activations/).
- Gradient descent variants: Batch gradient descent, Stochastic gradient descent (SGD), Mini-batch gradient descent.
- Gradient descent optimization algorithms: Momentum, Nesterov accelerated gradient, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam, AMSGrad. ([ref](https://arxiv.org/pdf/1609.04747.pdf))
- Parallelizing and distributing SGD: Hogwild!, Downpour SGD, Delay-tolerant Algorithms for SGD, TensorFlow, Elastic Averaging SGD.
- Additional strategies for optimizing SGD: Shuffling and Curriculum Learning, Batch normalization, Early Stopping, Gradient noise.
- Regularization techniques: L2/L1/Maxnorm regularization, dropout, batch and weight normalization, data augmentation, early stopping, gradient checking.
- Cross-validation: https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Common_types_of_cross-validation (hold-out, stratified k-fold,...)
- Capsule Networks. R-CNN. (time sequences) RNN and more modern LSTM. RBM. Echo-state networks. Inception modules.
- GANs.
- Transfer learning.

# Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL)

## RL/DRL Frameworks
Attempling to order by current popularity:
- rllab ([GitHub](https://github.com/rll/rllab)) ([readthedocs](http://rllab.readthedocs.io)) (officialy uses theano; in practice has some keras, tensorflow, torch, chainer...)
- Keras
   - https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec
   - https://github.com/haarnoja/sac
   - https://github.com/SoyGema/Startcraft_pysc2_minigames
- PyTorch
   - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
   - https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
- Torch
   - https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner
   - https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752 (originally in tensorflow)
- ChainerRL ([GitHub](https://github.com/chainer/chainerrl)) (API: Python)
- TensorForce ([GitHub](https://github.com/reinforceio/tensorforce)) (uses tensorflow)
- keras-rl ([GitHub](https://github.com/keras-rl/keras-rl)) (uses keras)
- https://github.com/geek-ai/MAgent (uses tensorflow)
- http://ray.readthedocs.io/en/latest/rllib.html (API: Python)
- http://burlap.cs.brown.edu/ (API: Java)

## RL/DRL Books
- Reinforcement Learning: An Introduction: http://incompleteideas.net/book/bookdraft2017nov5.pdf (Richard S. Sutton is father of RL)
- Andrew Ng thesis: www.cs.ubc.ca/~nando/550-2006/handouts/andrew-ng.pdf

## RL/DRL Gyms
- [OpenAI Gym](https://gym.openai.com) ([GitHub](https://github.com/openai/gym)) ([docs](https://gym.openai.com/docs/))
   - https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym
   - Most recent [robotics environments](https://gym.openai.com/envs/#robotics) use MuJoCo, but old examples have been re-implemented in [roboschool](https://blog.openai.com/roboschool/) which is visually the same but MIT License.
   - https://github.com/duckietown/gym-duckietown
   - https://github.com/arex18/rocket-lander
   - https://github.com/ppaquette/gym-doom
   - https://github.com/benelot/pybullet-gym
- https://github.com/mwydmuch/ViZDoom
- PySC2 ([GitHub](https://github.com/deepmind/pysc2)) (by DeepMind) (API: Python) (Blizzard StarCraft II Learning Environment (SC2LE) component)
- https://github.com/Microsoft/malmo
- https://github.com/nadavbh12/Retro-Learning-Environment
- https://github.com/twitter/torch-twrl
- Atari-net ?

## RL/DRL Baselines
- https://github.com/openai/baselines

## RL/DRL Techniques Misc
- Batch: REINFORCE, Deep Q-Network (DQN),  Expected-SARSA, True Online Temporal-Difference (TD), Double DQN, Truncated Natural Policy Gradient (TNPG), Trust Region Policy Optimization (TRPO), Reward-Weighted Regression, Relative Entropy Policy Search (REPS), Cross Entropy Method (CEM), Advantage-Actor-Critic (A2C), Asynchronous Advantage Actor-Critic (A3C), Actor-critic with Experience Replay (ACER), Actor Critic using Kronecker-Factored Trust Region (ACKTR), Generative Adversarial Imitation Learning (GAIL), Hindsight Experience Replay (HER), Proximal Policy Optimization (PPO, PPO1, PPO2), Ape-X Distributed Prioritized Experience Replay, Continuous DQN (CDQN or NAF), Dueling network DQN (Dueling DQN), Deep SARSA, Multi-Agent Deep Deterministic Policy Gradient (MADDPG).
- Online: Deep Determisitc Policy Gradient (DDPG).
- Experience Replay. 

# Evolutionary Algorithms (EA)
Only accounting those with same objective as RL.
- https://blog.openai.com/evolution-strategies
- https://eng.uber.com/deep-neuroevolution/
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    - https://github.com/CMA-ES/pycma
    - https://github.com/hardmaru/estool

# Misc Tools
- DLPaper2Code: Auto-generation of Code from Deep Learning Research Papers: https://arxiv.org/abs/1711.03543
- Tip: you can download the raw source of any arxiv paper. Click on the "Other formats" link, then click "Download source"
- http://www.arxiv-sanity.com/
