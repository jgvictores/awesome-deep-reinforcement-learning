# awesome-deep-reinforcement-learning

Curated list for Deep Reinforcement Learning (DRL): software frameworks, models, datasets, gyms, baselines...

To accomplish this, includes general Machine Learning (ML), Neural Networks (NN) and Deep Neural Networks (DNN) with many vision examples, and Reinforcement Learning (RL) with videogames/robotics examples. Some alternative Evolutionary Algorithms (EA) with similar objectives included too.

- [General Machine Learning (ML)](#general-machine-learning-ml)
   - [General ML Books / Papers](#general-ml-frameworks)
- [Neural Networks (NN) and Deep Neural Networks (DNN)](#neural-networks-nn-and-deep-neural-networks-dnn)
   - [NN/DNN Software Frameworks](#nndnn-software-frameworks)
   - [NN/DNN Models](#nndnn-models)
   - [NN/DNN Datasets](#nndnn-datasets)
   - [NN/DNN Benchmarks](#nndnn-benchmarks)
   - [NN/DNN Pretrained Models](#nndnn-pretrained-models)
   - [NN/DNN Techniques Misc](#nndnn-techniques-misc)
   - [NN/DNN Visualization](#nndnn-visualization)
- [Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL)](#reinforcement-learning-rl-and-deep-reinforcement-learning-drl)
   - [RL/DRL Software Frameworks](#rldrl-software-frameworks)
   - [RL/DRL Books](#rldrl-books)
   - [RL/DRL Gyms](#rldrl-gyms)
   - [RL/DRL Baselines](#rldrl-baselines)
   - [RL/DRL Techniques Misc](#rldrl-techniques-misc)
- [Evolutionary Algorithms (EA)](#evolutionary-algorithms-ea)
- [Misc tools](#misc-tools)
- [Similar pages](#similar-pages)

# General Machine Learning (ML)

## General ML Frameworks
- [scikit-learn](http://scikit-learn.org) (API: Python)
- [scikit-image](http://scikit-image.org) (API: Python)

## General ML Books / Papers
- Jake VanderPlas, "Python Data Science Handbook", 2017. ([Safari](http://proquest.safaribooksonline.com/book/programming/python/9781491912126)) (API: Python)
- https://github.com/terryum/awesome-deep-learning-papers#new-papers

# Neural Networks (NN) and Deep Neural Networks (DNN) 

## NN/DNN Software Frameworks
Here's a good overview [presentation](https://project.inria.fr/deeplearning/files/2016/05/DLFrameworks.pdf) ([permalink](https://github.com/jgvictores/awesome-deep-reinforcement-learning/blob/143a885cc10b4331b9b3fa3e1a9436d5325676af/doc/inria2017DLFrameworks.pdf)). There are several Docker images with several pre-installed software frameworks: [1](https://github.com/ufoym/deepo), [2](https://github.com/floydhub/dl-docker). Attempling to order software frameworks by current popularity:
- [Keras](https://keras.io) (layer over: TensorFlow, theano...) (API: Python) (support: Google). [wikipedia](https://en.wikipedia.org/wiki/Keras)
   - Examples/tutorials: [keras](https://github.com/keras-team/keras/blob/master/examples), [1](https://www.datacamp.com/community/tutorials/deep-learning-python), [2](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)
- [PyTorch](https://pytorch.org/) (API: Python) (support: Facebook AI Research).
   - Used internally by http://www.fast.ai/
- [Torch](http://torch.ch/) (API: Lua) (support: Facebook AI Research).
   - Multi-layer Recurrent Neural Networks (LSTM, GRU, RNN) for character-level language models: [1](https://github.com/karpathy/char-rnn)
- [TensorFlow](https://www.tensorflow.org) (low-level) (API: Python most stable, JavaScript, C++, Java...) (support: Google).
   - Tutorial: [style transfer](http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks)
   - Another NN wrapper: Tensorpack ([GitHub](https://github.com/ppwwyyxx/tensorpack))
- [Caffe](http://caffe.berkeleyvision.org/) (API: Python, Matlab) (support: Berkeley)
- [Chainer](http://www.chainer.org) ([GitHub](https://github.com/chainer/chainer)) (API: Python) (support: Preferred Networks)
   - Define-by-Run rather than Define-and-Run.
   - In addition to chainerrl below, there is also a chainercv: [1](https://github.com/chainer/chainercv)
- [Sonnet](https://deepmind.com/blog/open-sourcing-sonnet/) ([GitHub](https://github.com/deepmind/sonnet)) (layer over: TensorFlow) (API: Python) (support: DeepMind)
- [MXNet](https://mxnet.apache.org/) (API: Python, C++, Clojure, Julia, Perl, R, Scala) (support: Apache)
   - Tutorial: [1](https://github.com/llSourcell/MXNet)
- [Darknet](https://pjreddie.com/darknet) (API: C)
- [DL4J](https://deeplearning4j.org/) (API: Java)
- [OpenCV](https://www.opencv.org) now has some DNN: https://docs.opencv.org/3.3.0/d2/d58/tutorial_table_of_content_dnn.html
- [OpenNN](http://www.opennn.net) (API: C++)
- [PyBrain](http://www.pybrain.org) (API: Python)
- theano (very used, but down here because [MILA stopped developing](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ))
   - Still many tutorials: https://github.com/lisa-lab/DeepLearningTutorials

## NN/DNN Models
### Convolutional Models
Overviews: [1](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html), [2](http://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/), [3](https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5)

- Mask R-CNN (2017), Kaiming He et Al; Facebook AI Research (FAIR); "Mask R-CNN"; [arxiv](https://arxiv.org/abs/1703.06870). Refs: [1](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e), [2](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4).
- MobileNets (2017). Andrew Howard et Al; Google; "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"; [arxiv](https://arxiv.org/abs/1704.04861).
- FCIS (2017). "Fully Convolutional Instance-aware Semantic Segmentation". [arxiv](https://arxiv.org/abs/1611.07709). Coded in Caffe, released [MXNet](https://github.com/msracver/FCIS), port: [chainer](https://github.com/knorth55/chainer-fcis).
- DenseNets (2016). Gao Huang et Al; "Densely Connected Convolutional Networks"; [arxiv](https://arxiv.org/abs/1608.06993v1).
- YOLO9000 (2016). Joseph Redmond and Ali Farhadi; U Washington and Allen AI; "YOLO9000: Better, Faster, Stronger"; [arxiv](https://arxiv.org/abs/1612.08242). [Darknet](https://pjreddie.com/darknet/yolo), ports: [tensorflow](https://github.com/thtrieu/darkflow).
- SSD (2015). Wei Liu et Al; UNC, Zoox, Google, et Al; "SSD: Single Shot MultiBox Detector"; [arxiv](https://arxiv.org/abs/1512.02325). (orig: [caffe](https://github.com/intel/caffe/wiki/SSD:-Single-Shot-MultiBox-Detector))
- ResNet (2015). Kaiming He et Al; Microsoft Research; "Deep Residual Learning for Image Recognition"; [arxiv](https://arxiv.org/abs/1512.03385). Introduces Skip Connections ("gated units" or "gated recurrent units") and/or(?) "Residual Blocks" and heavy batch normalization. Variants: ResNet50, ResNet101, ResNet152 (correspond to number of layers). 25.5 million parameters.
- VGGNet (Sept 2014). Karen Simonyan et Al; Visual Geometry Group (Oxford); "Very Deep Convolutional Networks for Large-Scale Image Recognition"; [arxiv](https://arxiv.org/abs/1409.1556). Input: 224x224x3. Conv/pool and fully connected. Variants: VGG11, VGG13, VGG16, VGG19 (correspond to number of layers); with batch normalization. 138 million parameters; trained on 4 Titan Black GPUs for 2-3 weeks.
- GoogLeNet/InceptionV1 (Sept 2014). Christian Szegedy et Al; Google, UNC; "Going Deeper with Convolutions"; [arxiv](https://arxiv.org/abs/1409.4842). 22 layer deep CNN. Only 4-7 million parameters, via smaller convs. A more aggressive cropping approach than that of Krizhevsky. Batch normalization, image distortions, RMSprop. Uses 9 novel "Inception modules" (at each layer of a traditional ConvNet, you have to make a choice of whether to have a pooling operation or a conv operation as well as the choice of filter size; an Inception module performa all these operations in parallel), and no fully connected. Trained on CPU (estimated as weeks via GPU) implemented in DistBelief (closed-source predecessor of TensorFlow). Variants ([summary](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)): v1, v2, v4, resnet v1, resnet v2; v9 ([slides](http://lsun.cs.princeton.edu/slides/Christian.pdf)). Also see [Xception (2017)](https://arxiv.org/pdf/1610.02357.pdf) paper.
- NIN (2013). Min Lin et Al; NUSingapore; "Network In Network"; [arxiv](http://arxiv.org/abs/1312.4400). Provides inspiration for GoogLeNet.
- ZFNet (2013). Matthew D Zeiler and Rob Fergus; NYU; "Visualizing and Understanding Convolutional Networks"; [doi](https://doi.org/10.1007/978-3-319-10590-1_53), [arxiv](https://arxiv.org/abs/1311.2901). Similar to AlexNet, with well-justified finer tuning and visualization (namely Deconvolutional Network).
- AlexNet (2012). Alex Krizhevsky et Al; SuperVision (UToronto); "ImageNet Classification with Deep Convolutional Neural Networks"; [doi](https://doi.org/10.1145/3065386). In 224x224 color patches (and their horizontal reflections) from 256x256 color images; 5 conv, maxpool, 3 full; ReLU; SVD with momentum; dropout and data augmentation. 60-61 million parameters, split into 2 pipelines to enable 5-6 day GTX 580 GPU training (while CPU data augmentation).
- LeNet-5 (1998). Yann LeCun et Al; ATT now at Facebook AI Research; "Gradient-based learning applied to document recognition"; [doi](https://doi.org/10.1109/5.726791). In 32x32 grayscale; 7 layer (conv, pool, full...). 60 thousand parameters.

### Generative Models
- [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio) (2016). Aäron van den Oord et Al; DeepMind; "WaveNet: A Generative Model for Raw Audio"; [arxiv](https://arxiv.org/pdf/1609.03499.pdf). [wikipedia](https://en.wikipedia.org/wiki/WaveNet).
- Deep Dream. [caffe](https://github.com/google/deepdream)
- https://github.com/mlberkeley/Creative-Adversarial-Networks
- https://github.com/tkarras/progressive_growing_of_gans
- [CycleGAN](https://junyanz.github.io/CycleGAN/) (2017). Jun-Yan Zhu et Al; Berkeley; "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks". [torch](https://github.com/junyanz/CycleGAN) and migrated to [pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- Auto-Regressive Generative Models: PixelRNN, PixelCNN++... [ref](https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173)
- DCGAN (2015). Alec Radford, Luke Metz, Soumith Chintala; Indico Research, Facebook AI Research; "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"; [arxiv](https://arxiv.org/abs/1511.06434).
- GAN (2014). Ian J. Goodfellow et Al; Université de Montréal; "Generative Adversarial Nets"; [arxiv](https://arxiv.org/abs/1406.2661).

### Style Transfer Models
- https://github.com/ckmarkoh/neuralart_tensorflow
- https://github.com/luanfujun/deep-photo-styletransfer

### Recurrent Models
Can be trained via Back Propagation Through Time (BPTT). Cells include: SimpleRNN (commonly has TanH activation as second derivative decays slowly to 0), Gated Recurrent Units (GRU), Long short-term memory (LSTM), ConvLSTM2D, LSTM with peephole connection; [keras](https://keras.io/layers/recurrent/).
- Recurrent Neural Networks (RNN).
- Bidirectional RNN.
- Stateful RNN.

### Word Embedding Models
- GloVe (2014). Jeffrey Pennington et Al; Stanford; "GloVe: Global Vectors for Word Representation".
- [word2vec](https://code.google.com/archive/p/word2vec/).

### More Models
- Regression Networks (essentialy same, remove last activation and use some loss such as MSE rather than categorical cross-entropy).
- Autoencoders (AE), Variational Autoencoders (VAE), Denoising Autoencoders.
- Memory Networks. Use "Memory Units".
- Capsule Networks. Use "Capsules". [wikipedia](https://en.wikipedia.org/wiki/Capsule_neural_network)
- Echo-state networks.
- Restricted Boltzmann Machine (RBM).

## NN/DNN Datasets
Overview: [wikipedia](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)

### Images
- [MNIST](http://yann.lecun.com/exdb/mnist/): Handwritten digits, set of 70000 examples, is a subset of a larger set available from NIST.
- [ImageNet](http://www.image-net.org/): Project organized according to the WordNet hierarchy (22000 categories). Includes SIFT features, bounding boxes, attributes. Currently over 14 million images, 21841 cognitive synonyms (synsets) indexed, goal of +1000 images per synset.
   - ImageNet Large Visual Recognition Challenge (ILSVRC): Goal of 1000 categories using +100000 test images. E.g. LS-LOC
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (Visual Object Classes)
- [COCO](http://cocodataset.org) (Common Objects in Context): 2014, 2015, 2017. Includes classes and annotations.
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60000 32x32 colour images (selected from MIT TinyImages) in 10 classes, with 6000 images per class
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html): 60000 32x32 colour images (selected from MIT TinyImages) in 100 classes containing 600 images per class, grouped into 20 superclasses
- [SVHN](http://ufldl.stanford.edu/housenumbers/) (Street View House Numbers)
- [HICO](http://www-personal.umich.edu/~ywchao/hico/) (Humans Interacting with Common Objects)
- [PCam](https://github.com/basveeling/pcam) (PatchCamelyon)
- Robotics: [iCubWorld](https://robotology.github.io/iCubWorld/#datasets); where iCWT: 200 domestic objects in 20 categories (11 categories also in ILSVRC, rest in ImageNet).

### Text
- [text8](http://mattmahoney.net/dc/textdata.html): [text8.zip](http://mattmahoney.net/dc/text8.zip). more at [word2vec](https://code.google.com/archive/p/word2vec/).
- Sentiment Classification: [UMICH SI650](https://www.kaggle.com/c/si650winter11)
- Treebanks (text with part-of-speech (POS) tags): [wikipedia](https://en.wikipedia.org/wiki/Treebank), [Penn Treebank](https://web.archive.org/web/19970614160127/http://www.cis.upenn.edu/~treebank/)
- Facebook bAbI tasks: https://research.fb.com/downloads/babi

### Misc
- Kaggle: https://www.kaggle.com/datasets
- MIT: [MIT Places](http://places.csail.mit.edu/), [MIT Moments](http://moments.csail.mit.edu/)...
- UCI: https://archive.ics.uci.edu/ml/datasets.html

## NN/DNN Benchmarks
Overview: http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

## NN/DNN Pretrained Models
- Several pre-trained models: [keras web](https://keras.io/applications), [keras 1](https://github.com/keras-team/keras/tree/master/keras/applications), [keras 2](https://github.com/keras-team/keras-applications), [pytorch](https://pytorch.org/docs/stable/torchvision/models.html), [caffe](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- CIFAR-10 and CIFAR-100:
   - CNN trained on CIFAR-100 tutorial: [keras](https://andrewkruger.github.io/projects/2017-08-05-keras-convolutional-neural-network-for-cifar-100)
   - VGG16 trained on CIFAR-10 and CIFAR-100: [keras](https://github.com/geifmany/cifar-vgg) / [keras CIFAR-10 weights](https://drive.google.com/open?id=0B4odNGNGJ56qVW9JdkthbzBsX28) / [keras CIFAR-100 weights](https://drive.google.com/open?id=0B4odNGNGJ56qTEdnT1RjTU44Zms)
- ImageNet and ILSVRC:
   - VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, Xception trained on ImageNet: [keras by keras](https://github.com/keras-team/keras/tree/master/keras/applications) ([permalink](https://github.com/keras-team/keras/tree/e15533e6c725dca8c37a861aacb13ef149789433/keras/applications)) / [keras by kaggle](https://www.kaggle.com/keras) / [pytorch by kaggle](https://www.kaggle.com/pytorch)
   - VGG16 trained on ImageNet (tutorial): [keras](https://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/)
   - VGGNet, ResNet, Inception, and Xception trained on ImageNet (tutorial): [keras](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)
   - VGG16 trained on ILSVRC: [caffe by original VGG author](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) / ported (tutorials): [tensorflow](https://www.cs.toronto.edu/~frossard/post/vgg16/) / [keras](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) / [keras ImageNet weights](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc)
- word2vec: [gensim](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
- glove: http://nlp.stanford.edu/data/glove.6B.zip

## NN/DNN Techniques Misc
- Layers: Dense (aka Fully Connected), Convolutional (1D/2D/3D... [keras](https://keras.io/layers/convolutional), advanced: upsampling (e.g. in GANs), dilated causal (aka atrous)(e.g. in WaveNet)), Pooling (aka SubSampling)(1D/2D/3D)(Max, Average, Global Max, Global Average, Average with learnable weights per feature map... [keras](https://keras.io/layers/pooling)), Normalisation. Note: Keras implements activation functions, dropout, etc as layers.
- Weight initialization: pretrained (see [above section](#nndnn-pretrained)), zeros, ones, constant, normal random, uniform random, truncated normal, variance scaling, orthogonal, identity, normal/uniform as done by Yann LeCun, normal/uniform as done by Xavier Glorot, normal/uniform as done by Kaiming He. [keras](https://keras.io/initializers), [StackExchange](https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are)
- Activation functions: Linear, Sigmoid, Hard Sigmoid, Logit, Hyperbolic tangent (TanH), SoftSign, Rectified Linear Unit (ReLU), Leaky ReLU (LeakyReLU or LReLU), Parametrized or Parametric ReLU (PReLU), Thresholded ReLU (Thresholded ReLU), Exponential Linear Unit (ELU), Scaled ELU (SELU), SoftPlus, SoftMax, Swish. [wikipedia](https://en.wikipedia.org/wiki/Activation_function), [keras](https://keras.io/activations/), [keras (advanced)](https://keras.io/layers/advanced-activations/), [ref](https://towardsdatascience.com/deep-study-of-a-not-very-deep-neural-network-part-2-activation-functions-fd9bd8d406fc).
- Regularization techniques (reduce overfitting; may be applied to kernel (weight matrix), to bias vector, or to activity (activation of the layer output)): L1(lasso)/L2(ridge)/ElasticNet(L1/L2)/Maxnorm regularization ([keras](https://keras.io/regularizers/)), dropout, batch and weight normalization, Local Response Normalisation (LRN), data augmentation (image distortions, scale jittering...), early stopping, gradient checking.
- Optimizers: [keras](https://keras.io/optimizers/), [ref](https://arxiv.org/pdf/1609.04747.pdf)
   - Gradient descent variants: Batch gradient descent, Stochastic gradient descent (SGD), Mini-batch gradient descent.
   - Gradient descent optimization algorithms: Momentum, Nesterov accelerated gradient, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam, AMSGrad.
   - Parallelizing and distributing SGD: Hogwild!, Downpour SGD, Delay-tolerant Algorithms for SGD, TensorFlow, Elastic Averaging SGD.
   - Additional strategies for optimizing SGD: Shuffling and Curriculum Learning, Batch normalization, Early Stopping, Gradient noise.
- Error/loss functions: [keras](https://keras.io/losses)
   - Accuracy used for classification problems: binary accuracy (mean accuracy rate across all predictions for binary classification problems), categorical accuracy (mean accuracy rate across all predictions for multiclass classification problems), sparse categorical accuracy (useful for sparse targets), top k categorical accuracy (success when the target class is within the top k predictions provided).
   - Error loss (measures the difference between the values predicted and the values actually observed, can be used for regression): mean square error (MSE), root square error (RMSE), mean absolute error (MAE), mean absolute percentage error (MAPE), mean squared logarithmic error (MSLE).
   - Hinge: hinge loss, squared hinge loss, categorical hinge.
   - Class loss, used to calculate the cross-entropy for classification problems: binary cross-entropy (binary classification), categorical cross-entropy (multi-class classification), sparse categorical cross-entropy. [wikipedia](https://en.wikipedia.org/wiki/Cross_entropy)
   - Logarithm of the hyperbolic cosine of the prediction error (logcosh), kullback leibler divergence, poisson, cosine proximity.
- Metric functions: usually same type as error/loss functions, but used for evaluationg rather than training. [keras](https://keras.io/metrics)
- Cross-validation: hold-out, stratified k-fold. [wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Common_types_of_cross-validation).
- Transfer learning.

## NN/DNN Visualization
- [Quiver](https://github.com/keplr-io/quiver)
- Keras: [keras](https://keras.io/visualization/), [1](https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/)
- Tensorflow: [tensorflow online demo](http://playground.tensorflow.org)
- Caffe: [Netscope](http://ethereon.github.io/netscope) / [cnnvisualizer](https://github.com/metalbubble/cnnvisualizer)

# Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL)

## RL/DRL Software Frameworks
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
- https://github.com/thedimlebowski/Trading-Gym

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

# Similar pages
- https://github.com/tigerneil/awesome-deep-rl
- https://github.com/williamd4112/awesome-deep-reinforcement-learning
