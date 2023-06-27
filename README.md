# awesome-deep-reinforcement-learning

- [awesome-deep-reinforcement-learning](#awesome-deep-reinforcement-learning)
  - [General Machine Learning (ML)](#general-machine-learning-ml)
    - [General ML Software Frameworks](#general-ml-software-frameworks)
    - [General ML Books](#general-ml-books)
  - [Neural Networks (NN) and Deep Neural Networks (DNN)](#neural-networks-nn-and-deep-neural-networks-dnn)
    - [NN/DNN Software Frameworks](#nndnn-software-frameworks)
    - [NN/DNN Models](#nndnn-models)
      - [Image Object Segmentation, Localization, Detection Models](#image-object-segmentation-localization-detection-models)
      - [Image Segmentation Models](#image-segmentation-models)
      - [Image Detection Models](#image-detection-models)
      - [Image Classification Models](#image-classification-models)
      - [Graph/Manifold/Network Convolutional Models](#graphmanifoldnetwork-convolutional-models)
      - [Generative Models](#generative-models)
      - [Recurrent Models](#recurrent-models)
      - [Word Embedding Models](#word-embedding-models)
      - [More Models](#more-models)
    - [NN/DNN Datasets](#nndnn-datasets)
      - [Image Classification](#image-classification)
      - [Image Detection](#image-detection)
      - [Image Segmentation](#image-segmentation)
      - [Motion](#motion)
      - [Text](#text)
      - [Signal Separation](#signal-separation)
      - [Swiping (mobile text entry gesture typing)](#swiping-mobile-text-entry-gesture-typing)
    - [NN/DNN Benchmarks](#nndnn-benchmarks)
    - [NN/DNN Pretrained Models](#nndnn-pretrained-models)
    - [NN/DNN Techniques Misc](#nndnn-techniques-misc)
    - [NN/DNN Visualization and Explanation](#nndnn-visualization-and-explanation)
  - [Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL)](#reinforcement-learning-rl-and-deep-reinforcement-learning-drl)
    - [RL/DRL Algorithms](#rldrl-algorithms)
      - [RL/DRL algorithm classification adapted from Reinforcement Learning Specialization](#rldrl-algorithm-classification-adapted-from-reinforcement-learning-specialization)
      - [DRL algorithm classification adapted from CS285 at UC Berkeley](#drl-algorithm-classification-adapted-from-cs285-at-uc-berkeley)
      - [RL/DRL algorithm classification from OpenAI Spinning Up](#rldrl-algorithm-classification-from-openai-spinning-up)
      - [Just a random misc RL/DRL algorithms and techniques](#just-a-random-misc-rldrl-algorithms-and-techniques)
      - [Cool image](#cool-image)
    - [RL/DRL Algorithm Implementations and Software Frameworks](#rldrl-algorithm-implementations-and-software-frameworks)
    - [RL/DRL Environments](#rldrl-environments)
    - [RL/DRL Benchmarking](#rldrl-benchmarking)
    - [RL/DRL Books](#rldrl-books)
  - [Evolutionary Algorithms (EA)](#evolutionary-algorithms-ea)
  - [Misc Tools](#misc-tools)
  - [Similar pages](#similar-pages)

Curated list for Deep Reinforcement Learning (DRL): software frameworks, models, datasets, gyms, baselines...

To accomplish this, includes general Machine Learning (ML), Neural Networks (NN) and Deep Neural Networks (DNN) with many vision examples, and Reinforcement Learning (RL) with videogames/robotics examples. Some alternative Evolutionary Algorithms (EA) with similar objectives included too.

## General Machine Learning (ML)

### General ML Software Frameworks

- [scikit-learn](http://scikit-learn.org) (API: Python)
- [scikit-image](http://scikit-image.org) (API: Python)
- [microsoft/DirectML](https://github.com/microsoft/DirectML) (API: C++, Python)

### General ML Books

- Jake VanderPlas, "Python Data Science Handbook", 2017. [safari](http://proquest.safaribooksonline.com/book/programming/python/9781491912126)

## Neural Networks (NN) and Deep Neural Networks (DNN)

### NN/DNN Software Frameworks

- Overview: [presentation](https://project.inria.fr/deeplearning/files/2016/05/DLFrameworks.pdf) ([permalink](https://github.com/jgvictores/awesome-deep-reinforcement-learning/blob/143a885cc10b4331b9b3fa3e1a9436d5325676af/doc/inria2017DLFrameworks.pdf)).
- Docker images with several pre-installed software frameworks: [1](https://github.com/ufoym/deepo), [2](https://github.com/floydhub/dl-docker), [3](https://github.com/bethgelab/docker-deeplearning).
- Projects to port trained models from one software framework to another: [1](https://github.com/ysh329/deep-learning-model-convertor)

Attempling to order software frameworks by popularity (in practice should look at more aspects such as last updates, forks, etc):

- [pytorch/pytorch](https://github.com/pytorch/pytorch) [PyTorch](https://pytorch.org/) (API: Python) (support: Facebook AI Research). [![GitHub stars](https://img.shields.io/github/stars/pytorch/pytorch)](https://github.com/pytorch/pytorch/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/pytorch/pytorch?label=last%20update)
- [keras-team/keras](https://github.com/keras-team/keras) [Keras](https://keras.io) (layer over: TensorFlow, theano...) (API: Python) (support: Google). [wikipedia](https://en.wikipedia.org/wiki/Keras) [![GitHub stars](https://img.shields.io/github/stars/keras-team/keras)](https://github.com/keras-team/keras/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/keras-team/keras?label=last%20update)
  - Examples/tutorials: [keras](https://github.com/keras-team/keras/blob/master/examples), [1](https://www.datacamp.com/community/tutorials/deep-learning-python), [2](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)
  - Book: Antonio Gulli and Sujit Pal, "Deep Learning with Keras", 2017. [safari](https://proquest.safaribooksonline.com/book/programming/machine-learning/9781787128422)
  - Book: Mike Bernico, "Deep Learning Quick Reference", 2018. [safari](https://proquest.safaribooksonline.com/book/programming/machine-learning/9781788837996)
  - Used internally by <http://www.fast.ai>
- [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) [TensorFlow](https://www.tensorflow.org) (low-level) (API: Python most stable, JavaScript, C++, Java...) (support: Google). [![GitHub stars](https://img.shields.io/github/stars/tensorflow/tensorflow)](https://github.com/tensorflow/tensorflow/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/tensorflow/tensorflow?label=last%20update)
  - Tutorials: [1](https://medium.com/@tifa2up/image-classification-using-deep-neural-networks-a-beginner-friendly-approach-using-tensorflow-94b0a090ccd4)
- [flashlight/flashlight](https://github.com/flashlight/flashlight) [![GitHub stars](https://img.shields.io/github/stars/flashlight/flashlight)](https://github.com/flashlight/flashlight/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/flashlight/flashlight?label=last%20update)
- [https://github.com/janhuenermann/neurojs](https://github.com/janhuenermann/neurojs) [![GitHub stars](https://img.shields.io/github/stars/janhuenermann/neurojs)](https://github.com/janhuenermann/neurojs/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/janhuenermann/neurojs?label=last%20update)
- [ONNX](https://onnx.ai)
- [OpenCV](https://www.opencv.org) has DNN: <https://docs.opencv.org/3.3.0/d2/d58/tutorial_table_of_content_dnn.html>
- [Chainer](http://www.chainer.org) ([GitHub](https://github.com/chainer/chainer)) (API: Python) (support: Preferred Networks)
  - Define-by-Run rather than Define-and-Run.
  - In addition to chainerrl below, there is also a chainercv: [1](https://github.com/chainer/chainercv)
- [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) ([NVIDIA/DALI](https://github.com/NVIDIA/DALI)): A GPU-accelerated library containing highly optimized building blocks and an execution engine for data processing to accelerate deep learning training and inference applications.
- [Sonnet](https://deepmind.com/blog/open-sourcing-sonnet/) ([GitHub](https://github.com/deepmind/sonnet)) (layer over: TensorFlow) (API: Python) (support: DeepMind)
- [MXNet](https://mxnet.apache.org/) (API: Python, C++, Clojure, Julia, Perl, R, Scala) (support: Apache)
  - Tutorial: [1](https://github.com/llSourcell/MXNet)
- [Darknet](https://pjreddie.com/darknet) (API: C)
- [ml5](https://ml5js.org/) (API: JavaScript) (a tensorflow.js wrapper)
- [DL4J](https://deeplearning4j.org/) (API: Java)
- [oneapi-src/oneDNN](https://github.com/oneapi-src/oneDNN) (API: C++)
- [sony/nnabla](https://github.com/sony/nnabla) (API: C++)
- [Torch](http://torch.ch/) (API: Lua) (support: Facebook AI Research).
  - Multi-layer Recurrent Neural Networks (LSTM, GRU, RNN) for character-level language models: [1](https://github.com/karpathy/char-rnn)
- [jittor](https://github.com/Jittor/jittor) (API: Python)
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle): PArallel Distributed Deep LEarning
- [CoreML](https://developer.apple.com/documentation/coreml) (API: Objective-C) (support: Apple)
- Tensorpack ([GitHub](https://github.com/ppwwyyxx/tensorpack)) (a tensorflow wrapper)
- Ignite ([GitHub](https://github.com/pytorch/ignite)) (a pytorch wrapper)
- TransmogrifAI ([GitHub](https://github.com/salesforce/TransmogrifAI)) (API: Scala)
- tiny-dnn ([GitHub](https://github.com/tiny-dnn/tiny-dnn)) (API: C++ (C++14))
- [OpenNN](http://www.opennn.net) (API: C++)
- [PyBrain](http://www.pybrain.org) (API: Python)
- [Caffe](http://caffe.berkeleyvision.org/) (very used, but down here because [caffe2 merged into pytorch](https://caffe2.ai/blog/2018/05/02/Caffe2_PyTorch_1_0.html))
- theano (very used, but down here because [MILA stopped developing](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ))
  - Still many tutorials: <https://github.com/lisa-lab/DeepLearningTutorials>

### NN/DNN Models

#### Image Object Segmentation, Localization, Detection Models

Overviews: [1](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e). Taxonomy: [1](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf).

#### Image Segmentation Models

- Detectron (2018). Ross Girshick et Al; FAIR. [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron/) and [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
- FCIS (2017). "Fully Convolutional Instance-aware Semantic Segmentation". [arxiv](https://arxiv.org/abs/1611.07709). Coded in caffe but released in [mxnet](https://github.com/msracver/FCIS), port: [chainer](https://github.com/knorth55/chainer-fcis).
- U-Net (2015); Olaf Ronneberger et Al; "Convolutional Networks for Biomedical Image Segmentation"; [arxiv](https://arxiv.org/abs/1505.04597). [caffe](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

#### Image Detection Models

- YOLO (2015). Joseph Redmond et Al; U Washington, Allen AI, FAIR; "You Only Look Once: Unified, Real-Time Object Detection"; [arxiv](https://arxiv.org/abs/1506.02640). Variants: YOLO9000, YOLO v3... [Darknet](https://pjreddie.com/darknet/yolo), ports: [tensorflow](https://github.com/thtrieu/darkflow).
- SSD (2015). Wei Liu et Al; UNC, Zoox, Google, et Al; "SSD: Single Shot MultiBox Detector"; [arxiv](https://arxiv.org/abs/1512.02325). [caffe](https://github.com/intel/caffe/wiki/SSD:-Single-Shot-MultiBox-Detector)
- OverFeat (2015). Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann LeCun; NYU; "OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks"; [arxiv](https://arxiv.org/abs/1312.6229).
- R-CNN (2013). Ross Girshick et Al; Berkeley; "Rich feature hierarchies for accurate object detection and semantic segmentation"; [arxiv](https://arxiv.org/abs/1311.2524). Variants ([summary](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)): Fast R-CNN, Faster R-CNN, Mask R-CNN.

#### Image Classification Models

Overviews: [1](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html), [2](http://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/), [3](https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5)

- EfficientNets (2019). Mingxing Tan and Quoc V. Le; Google; "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"; [arxiv](https://arxiv.org/abs/1905.11946).
- MobileNets (2017). Andrew Howard et Al; Google; "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"; [arxiv](https://arxiv.org/abs/1704.04861).
- DenseNets (2017). Gao Huang et Al; "Densely Connected Convolutional Networks"; [arxiv](https://arxiv.org/abs/1608.06993v1). [torch](https://github.com/liuzhuang13/DenseNet) includes links to ports.
- ResNet (2015). Kaiming He et Al; Microsoft Research; "Deep Residual Learning for Image Recognition"; [arxiv](https://arxiv.org/abs/1512.03385). Introduces "Residual Blocks" via "Skip Connections" (some cite similarities with GRUs), and additionally uses heavy batch normalization. Variants: ResNet50, ResNet101, ResNet152 (correspond to number of layers). 25.5 million parameters.
- VGGNet (Sept 2014). Karen Simonyan, Andrew Zisserman; Visual Geometry Group (Oxford); "Very Deep Convolutional Networks for Large-Scale Image Recognition"; [arxiv](https://arxiv.org/abs/1409.1556). Input: 224x224x3. Conv/pool and fully connected. Variants: VGG11, VGG13, VGG16, VGG19 (correspond to number of layers); with batch normalization. 138 million parameters; trained on 4 Titan Black GPUs for 2-3 weeks.
- GoogLeNet/InceptionV1 (Sept 2014). Christian Szegedy et Al; Google, UNC; "Going Deeper with Convolutions"; [arxiv](https://arxiv.org/abs/1409.4842). 22 layer deep CNN. Only 4-7 million parameters, via smaller convs. A more aggressive cropping approach than that of Krizhevsky. Batch normalization, image distortions, RMSprop. Uses 9 novel "Inception modules" (at each layer of a traditional ConvNet, you have to make a choice of whether to have a pooling operation or a conv operation as well as the choice of filter size; an Inception module performa all these operations in parallel), and no fully connected. Trained on CPU (estimated as weeks via GPU) implemented in DistBelief (closed-source predecessor of TensorFlow). Variants ([summary](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)): v1, v2, v4, resnet v1, resnet v2; v9 ([slides](http://lsun.cs.princeton.edu/slides/Christian.pdf)). Also see [Xception (2017)](https://arxiv.org/pdf/1610.02357.pdf) paper.
- NIN (2013). Min Lin et Al; NUSingapore; "Network In Network"; [arxiv](http://arxiv.org/abs/1312.4400). Provides inspiration for GoogLeNet.
- ZFNet (2013). Matthew D Zeiler and Rob Fergus; NYU; "Visualizing and Understanding Convolutional Networks"; [doi](https://doi.org/10.1007/978-3-319-10590-1_53), [arxiv](https://arxiv.org/abs/1311.2901). Similar to AlexNet, with well-justified finer tuning and visualization (namely Deconvolutional Network).
- AlexNet (2012). Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton; SuperVision (UToronto); "ImageNet Classification with Deep Convolutional Neural Networks"; [doi](https://doi.org/10.1145/3065386). In 224x224 (227x227?) color patches (and their horizontal reflections) from 256x256 color images; 5 conv, maxpool, 3 full; ReLU; SVD with momentum; dropout and data augmentation. 60-61 million parameters, split into 2 pipelines to enable 5-6 day GTX 580 GPU training (while CPU data augmentation).
- LeNet-5 (1998). Yann LeCun et Al; ATT now at Facebook AI Research; "Gradient-based learning applied to document recognition"; [doi](https://doi.org/10.1109/5.726791). In 32x32 grayscale; 7 layer (conv, pool, full...). 60 thousand parameters.

#### Graph/Manifold/Network Convolutional Models

- [thunlp/GNNPapers](https://github.com/thunlp/GNNPapers)
- [Geometric deep learning](http://geometricdeeplearning.com)
- [chihming/awesome-network-embedding](https://github.com/chihming/awesome-network-embedding)
- [DLG](http://dgl.ai/): [dmlc/dgl](https://github.com/dmlc/dgl)
- "Signed Graph Convolutional Network" (ICDM 2018); [pytorch](https://github.com/benedekrozemberczki/SGCN)
- [tensorflow/gnn](https://github.com/tensorflow/gnn)

#### Generative Models

Tutorial: [pytorch](https://github.com/leongatys/GenerativeImageModellingWithDNNs)

- Auto-Regressive Generative Models: PixelRNN, PixelCNN++... [ref](https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173)
- Deep Dream. [caffe](https://github.com/google/deepdream)
- Style Transfer:
  - Tutorial: [tensorflow](http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks)
  - Fujun Luan et Al (2018), "Deep Painterly Harmonization"; [arxiv](https://arxiv.org/abs/1804.03189). [torch+matlab](https://github.com/luanfujun/deep-painterly-harmonization)
  - Deep Photo Style Transfer (2017). Fujun Luan et Al, "Deep Photo Style Transfer"; [arxiv](https://arxiv.org/abs/1703.07511). [torch+matlab](https://github.com/luanfujun/deep-photo-styletransfer)
  - Neuralart (2015). Leon A. Gatys et Al; "A Neural Algorithm of Artistic Style"; [arxiv](https://arxiv.org/abs/1508.06576). Uses base+style+target as inputs and optimizes for target via BFGS. [tensorflow](https://github.com/ckmarkoh/neuralart_tensorflow), [torch](https://github.com/jcjohnson/neural-style), keras [1](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py) [2](https://github.com/titu1994/Neural-Style-Transfer) [3](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-fun-with-deep-learning.md) [4](https://medium.com/mlreview/making-ai-art-with-style-transfer-using-keras-8bb5fa44b216)
- GANs:
  - [hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)
  - BigGAN (2018); "Large Scale GAN Training for High Fidelity Natural Image Synthesis"; [arxiv](https://arxiv.org/abs/1809.11096). [pytorch](https://github.com/AaronLeong/BigGAN-pytorch)
  - Terro Karas et Al (2018); NVIDIA; "Progressive Growing of GANs for Improved Quality, Stability, and Variation"; [arxiv](https://arxiv.org/abs/1710.10196). [tensorflow](https://github.com/tkarras/progressive_growing_of_gans)
  - CANs (2017). Ahmed Elgammal et Al; Berkeley; "CAN: Creative Adversarial Networks, Generating "Art" by Learning About Styles and Deviating from Style Norms"; [arxiv](https://arxiv.org/abs/1706.07068). [tensorflow](https://github.com/mlberkeley/Creative-Adversarial-Networks)
  - [CycleGAN](https://junyanz.github.io/CycleGAN/) (2017). Jun-Yan Zhu et Al; Berkeley; "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks". [torch](https://github.com/junyanz/CycleGAN) and migrated to [pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
  - DCGAN (2015). Alec Radford, Luke Metz, Soumith Chintala; Indico Research, Facebook AI Research; "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"; [arxiv](https://arxiv.org/abs/1511.06434).
  - GAN (2014). Ian J. Goodfellow et Al; Université de Montréal; "Generative Adversarial Nets"; [arxiv](https://arxiv.org/abs/1406.2661).
- Audio synthesis
  - [FTTNet](https://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/) (2018). Zeyu Jin et Al; "FFTNet: a Real-Time Speaker-Dependent Neural Vocoder". [pytorch](https://github.com/mozilla/FFTNet)
  - [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio) (2016). Aäron van den Oord et Al; DeepMind; "WaveNet: A Generative Model for Raw Audio"; [arxiv](https://arxiv.org/pdf/609.03499.pdf). [wikipedia](https://en.wikipedia.org/wiki/WaveNet).

#### Recurrent Models

Can be trained via Back Propagation Through Time (BPTT). Also see Connectionist Temporal Classification (CTC). Cells include: SimpleRNN (commonly has TanH activation as second derivative decays slowly to 0), Gated Recurrent Units (GRU), Long short-term memory (LSTM), ConvLSTM2D, LSTM with peephole connection; [keras](https://keras.io/layers/recurrent/).

- Recurrent Neural Networks (RNN).
- Bidirectional RNN.
- Stateful RNN.

#### Word Embedding Models

- BERT
- ELMo
- GloVe (2014). Jeffrey Pennington et Al; Stanford; "GloVe: Global Vectors for Word Representation".
- [word2vec](https://code.google.com/archive/p/word2vec/) (2013). Tomas Mikolov et Al; Google; "Distributed Representations of Words and Phrases and their Compositionality".

#### More Models

- Regression Networks (essentialy same, remove last activation and use some loss such as MSE rather than binary/categorical cross-entropy).
- Autoencoders (AE), Variational Autoencoders (VAE), Denoising Autoencoders.
  - Tutorials: [keras](https://blog.keras.io/building-autoencoders-in-keras.html), [keras](https://github.com/Puayny/Autoencoder-image-similarity)
  - Yunchen Pu et Al; "Variational Autoencoder for Deep Learning of Images, Labels and Captions"; [arxiv](https://arxiv.org/abs/1609.08976).
- Memory Networks. Use "Memory Units".
- Capsule Networks. Use "Capsules". [wikipedia](https://en.wikipedia.org/wiki/Capsule_neural_network)
- Echo-state networks.
- Restricted Boltzmann Machine (RBM).
- AutoML.

### NN/DNN Datasets

Lists of lists before citing the classics:

- [awesomedata/awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets)
- Wikipedia: <https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research>
- Google: <https://ai.google/tools/datasets>
- Kaggle: <https://www.kaggle.com/datasets>
- MIT: [MIT Places](http://places.csail.mit.edu/), [MIT Moments](http://moments.csail.mit.edu/)...
- UCI: <https://archive.ics.uci.edu/ml/datasets.html>
- Zillow: <https://www.zillow.com/research/data>
- Open Graph Benchmark: <https://ogb.stanford.edu>
- <https://data.world>
- <https://mbejda.github.io/>

#### Image Classification

- [MNIST](http://yann.lecun.com/exdb/mnist/): Handwritten digits, set of 70000 28x28 images, is a subset of a larger set available from NIST (and centered from its 32x32). Also see 2018's [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist).
- [ImageNet](http://www.image-net.org/): Project organized according to the WordNet hierarchy (22000 categories). Includes SIFT features, bounding boxes, attributes. Currently over 14 million images, 21841 cognitive synonyms (synsets) indexed, goal of +1000 images per synset.
  - ImageNet Large Visual Recognition Challenge (ILSVRC): Goal of 1000 categories using +100000 test images. E.g. LS-LOC
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (Visual Object Classes)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60000 32x32 colour images (selected from MIT TinyImages) in 10 classes, with 6000 images per class
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html): 60000 32x32 colour images (selected from MIT TinyImages) in 100 classes containing 600 images per class, grouped into 20 superclasses
- [MIT MM Stimuli](http://cvcl.mit.edu/MM/stimuli.html): Massive Memory (MM) Stimuli contains Unique Objects, State Pairs, State x Color Pairs...

#### Image Detection

- [SVHN](http://ufldl.stanford.edu/housenumbers/) (Street View House Numbers)
- [HICO](http://www-personal.umich.edu/~ywchao/hico/) (Humans Interacting with Common Objects)
- [Visual Genome](http://visualgenome.org/): Includes structured image concepts to language

#### Image Segmentation

- [COCO](http://cocodataset.org) (Common Objects in Context): 2014, 2015, 2017. Includes classes and annotations.

#### Motion

- KIT Motion-Language: <https://motion-annotation.humanoids.kit.edu/dataset>
- Sketches: [Quick Draw (Google)](https://github.com/googlecreativelab/quickdraw-dataset)
- Driving: <https://robotcar-dataset.robots.ox.ac.uk/datasets/>
- Robotics: [iCubWorld](https://robotology.github.io/iCubWorld/#datasets); where iCWT: 200 domestic objects in 20 categories (11 categories also in ILSVRC, rest in ImageNet). Also [muratkrty/iCub-camera-dataset](https://github.com/muratkrty/iCub-camera-dataset).
- [Kinetics (DeepMind)](https://deepmind.com/research/open-source/kinetics)
- [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/)

#### Text

- [text8](http://mattmahoney.net/dc/textdata.html): [text8.zip](http://mattmahoney.net/dc/text8.zip). more at [word2vec](https://code.google.com/archive/p/word2vec/).
- Sentiment Classification: [UMICH SI650](https://www.kaggle.com/c/si650winter11)
- Treebanks (text with part-of-speech (POS) tags): [wikipedia](https://en.wikipedia.org/wiki/Treebank), [Penn Treebank](https://web.archive.org/web/19970614160127/http://www.cis.upenn.edu/~treebank/)
- Facebook bAbI tasks: <https://research.fb.com/downloads/babi>

#### Signal Separation

- SigSep: <https://sigsep.github.io/datasets/>

#### Swiping (mobile text entry gesture typing)

- How-We-Swipe Shape-writing Dataset: <https://osf.io/sj67f/>, [DOI: 10.1145/3447526.3472059](https://www.doi.org/10.1145/3447526.3472059)

### NN/DNN Benchmarks

- <https://benchmarks.ai>
- <http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html>
- <https://martin-thoma.com/sota/#computer-vision>
- <https://robust.vision/benchmark>
- [brain-research/realistic-ssl-evaluation](https://github.com/brain-research/realistic-ssl-evaluation)

### NN/DNN Pretrained Models

- Several pre-trained models: [keras web](https://keras.io/applications), [keras 1](https://github.com/keras-team/keras/tree/master/keras/applications), [keras 2](https://github.com/keras-team/keras-applications), [pytorch](https://pytorch.org/docs/stable/torchvision/models.html), [caffe](https://github.com/BVLC/caffe/wiki/Model-Zoo), [ONNX](https://github.com/onnx/models) (pytorch/caffe2).
- CIFAR-10 and CIFAR-100:
  - CNN trained on CIFAR-100 tutorial: [keras](https://andrewkruger.github.io/projects/2017-08-05-keras-convolutional-neural-network-for-cifar-100)
  - VGG16 trained on CIFAR-10 and CIFAR-100: [keras](https://github.com/geifmany/cifar-vgg) / [keras CIFAR-10 weights](https://drive.google.com/open?id=0B4odNGNGJ56qVW9JdkthbzBsX28) / [keras CIFAR-100 weights](https://drive.google.com/open?id=0B4odNGNGJ56qTEdnT1RjTU44Zms)
- ImageNet and ILSVRC:
  - VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, Xception trained on ImageNet: [keras by keras](https://github.com/keras-team/keras/tree/master/keras/applications) ([permalink](https://github.com/keras-team/keras/tree/e15533e6c725dca8c37a861aacb13ef149789433/keras/applications)) / [keras by kaggle](https://www.kaggle.com/keras) / [pytorch by kaggle](https://www.kaggle.com/pytorch)
  - VGG16 trained on ImageNet (tutorial): [keras](https://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/)
  - VGGNet, ResNet, Inception, and Xception trained on ImageNet (tutorial): [keras](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)
  - VGG16 trained on ILSVRC: [caffe by original VGG author](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) / ported (tutorials): [tensorflow](https://www.cs.toronto.edu/~frossard/post/vgg16/) / [keras](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) / [keras ImageNet weights](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc)
- word2vec: [gensim](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
- glove: <http://nlp.stanford.edu/data/glove.6B.zip>

### NN/DNN Techniques Misc

- Layers: Dense (aka Fully Connected), Convolutional (1D/2D/3D... [keras](https://keras.io/layers/convolutional), advanced: upsampling (e.g. in GANs), dilated causal (aka atrous)(e.g. in WaveNet)), Pooling (aka SubSampling)(1D/2D/3D)(Max, Average, Global Max, Global Average, Average with learnable weights per feature map... [keras](https://keras.io/layers/pooling)), Normalisation. Note: Keras implements activation functions, dropout, etc as layers.
- Weight initialization: pretrained (see [above section](#nndnn-pretrained)), zeros, ones, constant, normal random, uniform random, truncated normal, variance scaling, orthogonal, identity, normal/uniform as done by Yann LeCun, normal/uniform as done by Xavier Glorot, normal/uniform as done by Kaiming He. [keras](https://keras.io/initializers), [StackExchange](https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are)
- Activation functions: Linear, Sigmoid, Hard Sigmoid, Logit, Hyperbolic tangent (TanH), SoftSign, Rectified Linear Unit (ReLU), Leaky ReLU (LeakyReLU or LReLU), Parametrized or Parametric ReLU (PReLU), Thresholded ReLU (Thresholded ReLU), Exponential Linear Unit (ELU), Scaled ELU (SELU), SoftPlus, SoftMax, Swish. [wikipedia](https://en.wikipedia.org/wiki/Activation_function), [keras](https://keras.io/activations/), [keras (advanced)](https://keras.io/layers/advanced-activations/), [ref](https://towardsdatascience.com/deep-study-of-a-not-very-deep-neural-network-part-2-activation-functions-fd9bd8d406fc).
- Regularization techniques (reduce overfitting and/or control the complexity of model; may be applied to kernel (weight matrix), to bias vector, or to activity (activation of the layer output)): L1(lasso)/L2(ridge)/ElasticNet(L1/L2)/Maxnorm regularization ([keras](https://keras.io/regularizers/)), dropout, batch and weight normalization, Local Response Normalisation (LRN), data augmentation (image distortions, scale jittering...), early stopping, gradient checking.
- Optimizers: [keras](https://keras.io/optimizers/), [ref](https://arxiv.org/pdf/1609.04747.pdf)
  - Gradient descent variants: Batch gradient descent, Stochastic gradient descent (SGD), Mini-batch gradient descent.
  - Gradient descent optimization algorithms: Momentum, Nesterov accelerated gradient, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam, AMSGrad, Eve.
  - Parallelizing and distributing SGD: Hogwild!, Downpour SGD, Delay-tolerant Algorithms for SGD, TensorFlow, Elastic Averaging SGD.
  - Additional strategies for optimizing SGD: Shuffling and Curriculum Learning, Batch normalization, Early Stopping, Gradient noise.
  - Broyden-Fletcher-Goldfarb-Shanno (BFGS)
  - Gradient-free: [facebookresearch/nevergrad](https://github.com/facebookresearch/nevergrad)
- Error/loss functions: [keras](https://keras.io/losses)
  - Accuracy used for classification problems: binary accuracy (mean accuracy rate across all predictions for binary classification problems), categorical accuracy (mean accuracy rate across all predictions for multiclass classification problems), sparse categorical accuracy (useful for sparse targets), top k categorical accuracy (success when the target class is within the top k predictions provided).
  - Error loss (measures the difference between the values predicted and the values actually observed, can be used for regression): mean square error (MSE), root square error (RMSE), mean absolute error (MAE), mean absolute percentage error (MAPE), mean squared logarithmic error (MSLE).
  - Hinge: hinge loss, squared hinge loss, categorical hinge.
  - Class loss, used to calculate the cross-entropy for classification problems: binary cross-entropy (binary classification), categorical cross-entropy (multi-class classification), sparse categorical cross-entropy. [wikipedia](https://en.wikipedia.org/wiki/Cross_entropy)
  - Logarithm of the hyperbolic cosine of the prediction error (logcosh), kullback leibler divergence, poisson, cosine proximity.
- Metric functions: usually same type as error/loss functions, but used for evaluationg rather than training. [keras](https://keras.io/metrics)
- Cross-validation: hold-out, stratified k-fold. [wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Common_types_of_cross-validation).
- Transfer learning. [tensorflow](https://github.com/mluogh/transfer-learning), [keras](https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/)

### NN/DNN Visualization and Explanation

- [tensorboard](https://www.tensorflow.org/tensorboard)
- [tensorboardX](https://github.com/lanpa/tensorboardX): tensorboard for pytorch, chainer, mxnet, numpy...
- Keras: [keras](https://keras.io/visualization/), [1](https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/), [2](https://github.com/keplr-io/quiver), [3](https://raghakot.github.io/keras-vis/), [4](https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras)
- Tensorflow: [tensorflow online demo](http://playground.tensorflow.org)
- Pytorch: [loss-landscape](https://github.com/tomgoldstein/loss-landscape), [gandissect](https://github.com/CSAILVision/gandissect)
- Caffe: [netscope](http://ethereon.github.io/netscope) / [cnnvisualizer](https://github.com/metalbubble/cnnvisualizer)
- SHAP (SHapley Additive exPlanations): [slundberg/shap](https://github.com/slundberg/shap)
- XAI (An eXplainability toolbox for machine learning): [EthicalML/xai](https://github.com/EthicalML/xai)

## Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL)

### RL/DRL Algorithms

#### RL/DRL algorithm classification adapted from Reinforcement Learning Specialization

Classification of RL algorithms adapted from [Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning) (Martha & Adam White, from University of Alberta and Alberta Machine Intelligence Institute, on Coursera, 2019-20). Note that another major separation is off/on policy RL algorithms. DRL methods would fit into function approximators.

```text
+-- Tablular Methods
|   +-- Average Reward (e.g. for Continuing Tasks a.k.a. Infinite Horizon Case)
|   |   +-- Continuous Action Space
|   |   |   +-- Gaussian Actor-Critic
|   |   +-- Discrete Action Space
|   |       +-- Softmax Actor-Critic
|   |       +-- Differential Semi-Gradient SARSA
|   +-- Not using Average Reward (e.g. for Episodic Tasks a.k.a. Finite Horizon Case)
|       +-- Learn at each time step
|       |   +-- Control Problem
|       |   |   +-- Expected SARSA
|       |   |   +-- Q-Learning
|       |   |   +-- SARSA
|       |   +-- Not a Control Problem
|       |       +-- Semi-Gradient TD
|       +-- Not learn at each time step
|           +-- Gradient Monte Carlo
+-- Function Approximator Methods
    +-- Access to a model (model-based, part 1/2)
    |   +-- Control Problem
    |   |   +-- Value Iteration
    |   |   +-- Policy Iteration
    |   +-- Not a Control Problem
    |   |   +-- Iterative Policy Evaluation
    +-- No access to a model
        +-- Will learn a model (model-based, part 2/2)
        |   +-- Q-Planning
        |   +-- Dyna-Q+
        |   +-- Dyna-Q
        +-- Model-free
            +-- Learn at each time step
            |   +-- Control Problem
            |   |   +-- Q-Learning
            |   |   +-- Expected SARSA
            |   |   +-- SARSA
            |   +-- Not a Control Problem
            |       +-- TD
            +-- Not learn at each time step
                +-- Control Problem
                |   +-- eplsilon-soft Monte Carlo
                |   +-- Exploring starts Monte Carlo
                +-- Not a Control Problem
                    +-- Off-Policy Monte Carlo
                    +-- Monte Carlo Prediction
```

#### DRL algorithm classification adapted from CS285 at UC Berkeley

DRL algorithm classification adapted from [Deep Reinforcement Learning CS 285 at UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse), Sergey Levine, [Fall 2020](http://rail.eecs.berkeley.edu/deeprlcourse-fa20/), Lecture 4.

1. Policy Gradients
2. Value-based
3. Actor-critic
4. Model-based RL

#### RL/DRL algorithm classification from OpenAI Spinning Up

- [Part 2: Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) <- Rendered from <https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/docs/spinningup/rl_intro2.rst>

#### Just a random misc RL/DRL algorithms and techniques

REINFORCE (on-policy policy gradient; Williams, 1992), Deep Q-Network (DQN), Expected-SARSA, True Online Temporal-Difference (TD), Double DQN, Truncated Natural Policy Gradient (TNPG), Trust Region Policy Optimization (TRPO), Reward-Weighted Regression, Relative Entropy Policy Search (REPS), Cross Entropy Method (CEM), Advantage-Actor-Critic (A2C), Asynchronous Advantage Actor-Critic (A3C), Actor-critic with Experience Replay (ACER), Actor Critic using Kronecker-Factored Trust Region (ACKTR), Generative Adversarial Imitation Learning (GAIL), Hindsight Experience Replay (HER), Proximal Policy Optimization (PPO, PPO1, PPO2), Ape-X Distributed Prioritized Experience Replay, Continuous DQN (CDQN or NAF), Dueling network DQN (Dueling DQN), Deep SARSA, Multi-Agent Deep Deterministic Policy Gradient (MADDPG), Deep Deterministic Policy Gradient (DDPG), Truncated Quantile Critics (TQC), Quantile Regression DQN (QR-DQN).

#### Cool image

![](https://github.com/hijkzzz/deep-reinforcement-learning-notes/blob/603d5d9ba0975df9f164b929627593701404e48f/.gitbook/assets/all-1.png?raw=true)

### RL/DRL Algorithm Implementations and Software Frameworks

Attempting to order by popularity (in practice should look at more aspects such as last updates, forks, etc):

- RLlib (part of Ray): [ray-project/ray](https://github.com/ray-project/ray) ([readthedocs (rllib)](http://ray.readthedocs.io/en/latest/rllib.html)) [![GitHub stars](https://img.shields.io/github/stars/ray-project/ray)](https://github.com/ray-project/ray/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/ray-project/ray?label=last%20update) (Ray total) (also covers multiagent)
- [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents) (Environments, Algorithms) (includes design of environments) [![GitHub stars](https://img.shields.io/github/stars/Unity-Technologies/ml-agents)](https://github.com/Unity-Technologies/ml-agents/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/Unity-Technologies/ml-agents?label=last%20update)
- [google/dopamine](https://github.com/google/dopamine) (uses jax, tensorflow, keras) [![GitHub stars](https://img.shields.io/github/stars/google/dopamine)](https://github.com/google/dopamine/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/google/dopamine?label=last%20update)
- [keras-rl/keras-rl](https://github.com/keras-rl/keras-rl) (uses keras) [![GitHub stars](https://img.shields.io/github/stars/keras-rl/keras-rl)](https://github.com/keras-rl/keras-rl/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/keras-rl/keras-rl?label=last%20update)
  - [SoyGema/Startcraft_pysc2_minigames](https://github.com/SoyGema/Startcraft_pysc2_minigames)
- [thu-ml/tianshou](https://github.com/thu-ml/tianshou) (<https://tianshou.readthedocs.io>) [![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/thu-ml/tianshou?label=last%20update)
- [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (advanced from [hill-a/stable-baselines](https://github.com/hill-a/stable-baselines) fork of [openai/baselines](https://github.com/openai/baselines)) [![GitHub stars](https://img.shields.io/github/stars/DLR-RM/stable-baselines3)](https://github.com/DLR-RM/stable-baselines3/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/DLR-RM/stable-baselines3?label=last%20update)
- [https://github.com/janhuenermann/neurojs](https://github.com/janhuenermann/neurojs) [![GitHub stars](https://img.shields.io/github/stars/janhuenermann/neurojs)](https://github.com/janhuenermann/neurojs/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/janhuenermann/neurojs?label=last%20update)
- [deepmind/open_spiel](https://github.com/deepmind/open_spiel) (uses some tensorflow) [![GitHub stars](https://img.shields.io/github/stars/deepmind/open_spiel)](https://github.com/deepmind/open_spiel/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/deepmind/open_spiel?label=last%20update)
- [reinforceio/tensorforce](https://github.com/reinforceio/tensorforce) (uses tensorflow) [![GitHub stars](https://img.shields.io/github/stars/reinforceio/tensorforce)](https://github.com/reinforceio/tensorforce/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/reinforceio/tensorforce?label=last%20update)
- [deepmind/trfl](https://github.com/deepmind/trfl) (uses tensorflow) [![GitHub stars](https://img.shields.io/github/stars/deepmind/trfl)](https://github.com/deepmind/trfl/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/deepmind/trfl?label=last%20update)
- [catalyst-team/catalyst](https://github.com/catalyst-team/catalyst) [![GitHub stars](https://img.shields.io/github/stars/catalyst-team/catalyst)](https://github.com/catalyst-team/catalyst/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/catalyst-team/catalyst?label=last%20update)
- [deepmind/acme](https://github.com/deepmind/acme) [![GitHub stars](https://img.shields.io/github/stars/deepmind/acme)](https://github.com/deepmind/acme/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/deepmind/acme?label=last%20update)
- [rll/rllab](https://github.com/rll/rllab) ([readthedocs](http://rllab.readthedocs.io)) (officialy uses theano; in practice has some keras, tensorflow, torch, chainer...) [![GitHub stars](https://img.shields.io/github/stars/rll/rllab)](https://github.com/rll/rllab/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/rll/rllab?label=last%20update)
- TF-Agents: [tensorflow/agents](https://github.com/tensorflow/agents) (uses tensorflow) [![GitHub stars](https://img.shields.io/github/stars/tensorflow/agents)](https://github.com/tensorflow/agents/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/tensorflow/agents?label=last%20update)
- [astooke/rlpyt](https://github.com/astooke/rlpyt) (uses pytorch) [![GitHub stars](https://img.shields.io/github/stars/astooke/rlpyt)](https://github.com/astooke/rlpyt/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/astooke/rlpyt?label=last%20update)
- [rail-berkeley/rlkit](https://github.com/rail-berkeley/rlkit) [![GitHub stars](https://img.shields.io/github/stars/rail-berkeley/rlkit)](https://github.com/rail-berkeley/rlkit/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/rail-berkeley/rlkit?label=last%20update)
- [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) [![GitHub stars](https://img.shields.io/github/stars/vwxyzjn/cleanrl)](https://github.com/vwxyzjn/cleanrl/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/vwxyzjn/cleanrl?label=last%20update)
- [oxwhirl/pymarl](https://github.com/oxwhirl/pymarl) (support: <http://whirl.cs.ox.ac.uk>): deep multi-agent reinforcement learning [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/oxwhirl/pymarl?label=last%20update)
- [deepmind/bsuite](https://github.com/deepmind/bsuite) (Environments, Algorithm Implementations, Benchmarking) [![GitHub stars](https://img.shields.io/github/stars/deepmind/bsuite)](https://github.com/deepmind/bsuite/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/deepmind/bsuite?label=last%20update)
- [chainer/chainerrl](https://github.com/chainer/chainerrl) (API: Python) [![GitHub stars](https://img.shields.io/github/stars/chainer/chainerrl)](https://github.com/chainer/chainerrl/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/chainer/chainerrl?label=last%20update)
- [facebookresearch/rl](https://github.com/facebookresearch/rl) (uses pytorch) [![GitHub stars](https://img.shields.io/github/stars/facebookresearch/rl)](https://github.com/facebookresearch/rl/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/facebookresearch/rl?label=last%20update)
- [MushroomRL/mushroom-rl](https://github.com/MushroomRL/mushroom-rl) [![GitHub stars](https://img.shields.io/github/stars/MushroomRL/mushroom-rl)](https://github.com/MushroomRL/mushroom-rl/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/MushroomRL/mushroom-rl?label=last%20update)
- [SurrealAI/surreal](https://github.com/SurrealAI/surreal) (API: Python) (support: Stanford Vision and Learning Lab). [![GitHub stars](https://img.shields.io/github/stars/SurrealAI/surreal)](https://github.com/SurrealAI/surreal/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/SurrealAI/surreal?label=last%20update)
- [medipixel/rl_algorithms](https://github.com/medipixel/rl_algorithms) [![GitHub stars](https://img.shields.io/github/stars/medipixel/rl_algorithms)](https://github.com/medipixel/rl_algorithms/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/medipixel/rl_algorithms?label=last%20update)
- [ikostrikov/jaxrl2](https://github.com/ikostrikov/jaxrl2) [![GitHub stars](https://img.shields.io/github/stars/ikostrikov/jaxrl2)](https://github.com/ikostrikov/jaxrl2/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/ikostrikov/jaxrl2?label=last%20update)
  - [ikostrikov/jaxrl](https://github.com/ikostrikov/jaxrl) (uses JAX) [![GitHub stars](https://img.shields.io/github/stars/ikostrikov/jaxrl)](https://github.com/ikostrikov/jaxrl/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/ikostrikov/jaxrl?label=last%20update)
- [tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL) "High-quality single-file implementations of SOTA Offline RL algorithms: AWAC, BC, CQL, DT, EDAC, IQL, SAC-N, TD3+BC" [![GitHub stars](https://img.shields.io/github/stars/tinkoff-ai/CORL)](https://github.com/tinkoff-ai/CORL/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/tinkoff-ai/CORL?label=last%20update)
- [learnables/cherry](https://github.com/learnables/cherry) (API: Python) (layer over pytorch) [![GitHub stars](https://img.shields.io/github/stars/learnables/cherry)](https://github.com/learnables/cherry/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/learnables/cherry?label=last%20update)
- [trackmania-rl/tmrl](https://github.com/trackmania-rl/tmrl) [![GitHub stars](https://img.shields.io/github/stars/trackmania-rl/tmrl)](https://github.com/trackmania-rl/tmrl/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/trackmania-rl/tmrl?label=last%20update)
- [ethanluoyc/magi](https://github.com/ethanluoyc/magi) (uses JAX) [![GitHub stars](https://img.shields.io/github/stars/ethanluoyc/magi)](https://github.com/ethanluoyc/magi/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/ethanluoyc/magi?label=last%20update)
- [RL-Glue](https://sites.google.com/a/rl-community.org/rl-glue/Home/rl-glue) ([Google Code Archive](https://code.google.com/archive/p/rl-glue-ext/wikis/RLGlueCore.wiki)) (API: C/C++, Java, Matlab, Python, Lisp) (support: Alberta)
- <http://burlap.cs.brown.edu/> (API: Java)

Lower level:

- [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) [![GitHub stars](https://img.shields.io/github/stars/tensorflow/tensorflow)](https://github.com/tensorflow/tensorflow/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/tensorflow/tensorflow?label=last%20update)
- [pytorch/pytorch](https://github.com/pytorch/pytorch) <https://pytorch.org> [![GitHub stars](https://img.shields.io/github/stars/pytorch/pytorch)](https://github.com/pytorch/pytorch/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/pytorch/pytorch?label=last%20update)
  - <https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html>
- [keras-team/keras](https://github.com/keras-team/keras) <https://keras.io/> [![GitHub stars](https://img.shields.io/github/stars/keras-team/keras)](https://github.com/keras-team/keras/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/keras-team/keras?label=last%20update)
- [google/jax](https://github.com/google/jax): Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more. [![GitHub stars](https://img.shields.io/github/stars/google/jax)](https://github.com/google/jax/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/google/jax?label=last%20update)

Specific to Model-based:

- [facebookresearch/mbrl-lib](https://github.com/facebookresearch/mbrl-lib): Library for Model Based RL. [![GitHub stars](https://img.shields.io/github/stars/facebookresearch/mbrl-lib)](https://github.com/facebookresearch/mbrl-lib/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/facebookresearch/mbrl-lib?label=last%20update)

For specific algorithms (e.g. original paper implementations):

- [haarnoja/sac](https://github.com/haarnoja/sac)
- [Asap7772/PTR](https://github.com/Asap7772/PTR) Pre-Training for Robots: Leveraging Diverse Multitask Data via Offline Reinforcement Learning
- [ikostrikov/pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

Tutorials/education (typically from lower level):

- [openai/spinningup](https://github.com/openai/spinningup) (<https://spinningup.openai.com>, educational, uses pytorch updated from tensorflow) [![GitHub stars](https://img.shields.io/github/stars/openai/spinningup)](https://github.com/openai/spinningup/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/openai/spinningup?label=last%20update)
- [qfettes/DeepRL-Tutorials](https://github.com/qfettes/DeepRL-Tutorials) (uses pytorch) [![GitHub stars](https://img.shields.io/github/stars/qfettes/DeepRL-Tutorials)](https://github.com/qfettes/DeepRL-Tutorials/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/qfettes/DeepRL-Tutorials?label=last%20update)
- <https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec> (uses keras)
- <https://medium.com/@jonathan_hui/rl-reinforcement-learning-algorithms-quick-overview-6bf69736694d>

Comparison:

- <https://winder.ai/a-comparison-of-reinforcement-learning-frameworks-dopamine-rllib-keras-rl-coach-trfl-tensorforce-coach-and-more/>

### RL/DRL Environments

- [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (<https://gymnasium.farama.org>) [![GitHub stars](https://img.shields.io/github/stars/Farama-Foundation/Gymnasium)](https://github.com/Farama-Foundation/Gymnasium/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/Farama-Foundation/Gymnasium?label=last%20update). ~~DEPRECATED: [openai/gym](https://github.com/openai/gym), <https://gym.openai.com>, <https://gym.openai.com/docs/>~~
  - <https://github.com/Farama-Foundation/Gymnasium/network/dependents>
  - [Farama-Foundation/Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
  - [Farama-Foundation/Minigrid](https://github.com/Farama-Foundation/Minigrid)
  - [Farama-Foundation/MiniWorld](https://github.com/Farama-Foundation/MiniWorld)
  - [Farama-Foundation/ViZDoom](https://github.com/Farama-Foundation/ViZDoom) (was [mwydmuch/ViZDoom](https://github.com/mwydmuch/ViZDoom))
  - <https://gymnasium.farama.org/environments/third_party_environments>
  - ~~DEPRECATED: [openai/roboschool](https://github.com/openai/roboschool)~~
  - [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents) (Environments, Algorithms) (includes design of environments)
  - [Unity-Technologies/obstacle-tower-env](https://github.com/Unity-Technologies/obstacle-tower-env)
  - [LucasAlegre/sumo-rl](https://github.com/LucasAlegre/sumo-rl)
  - [qgallouedec/panda-gym](https://github.com/qgallouedec/panda-gym)
  - [NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
  - [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)
  - [utiasDSL/safe-control-gym](https://github.com/utiasDSL/safe-control-gym)
  - [deepmind/bsuite](https://github.com/deepmind/bsuite) (Environments, Algorithms, Benchmarking)
  - [openai/gym-soccer](https://github.com/openai/gym-soccer)
  - [erlerobot/gym-gazebo](https://github.com/erlerobot/gym-gazebo)
  - [robotology/gym-ignition](https://github.com/robotology/gym-ignition)
  - [dartsim/gym-dart](https://github.com/dartsim/gym-dart)
  - [Roboy/gym-roboy](https://github.com/Roboy/gym-roboy)
  - [kngwyu/mujoco-maze](https://github.com/kngwyu/mujoco-maze)
  - [Improbable-AI/walk-these-ways](https://github.com/Improbable-AI/walk-these-ways)
  - [ucuapps/modelicagym](https://github.com/ucuapps/modelicagym)
  - [openai/safety-gym](https://github.com/openai/safety-gym)
  - [openai/retro](https://github.com/openai/retro)
  - [deepmind/pysc2](https://github.com/deepmind/pysc2) (by DeepMind) (Blizzard StarCraft II Learning Environment (SC2LE) component)
  - [benelot/pybullet-gym](https://github.com/benelot/pybullet-gym)
  - [Healthcare-Robotics/assistive-gym](https://github.com/Healthcare-Robotics/assistive-gym)
  - [Microsoft/malmo](https://github.com/Microsoft/malmo)
  - [nadavbh12/Retro-Learning-Environment](https://github.com/nadavbh12/Retro-Learning-Environment)
  - [twitter/torch-twrl](https://github.com/twitter/torch-twrl)
  - [duckietown/gym-duckietown](https://github.com/duckietown/gym-duckietown)
  - [arex18/rocket-lander](https://github.com/arex18/rocket-lander)
  - [ppaquette/gym-doom](https://github.com/ppaquette/gym-doom)
  - [eleurent/highway-env](https://github.com/eleurent/highway-env)
  - [thedimlebowski/Trading-Gym](https://github.com/thedimlebowski/Trading-Gym)
  - [denisyarats/dmc2gym](https://github.com/denisyarats/dmc2gym)
  - [minerllabs/minerl](https://github.com/minerllabs/minerl)
  - [eugenevinitsky/sequential_social_dilemma_games](https://github.com/eugenevinitsky/sequential_social_dilemma_games)
  - [facebookresearch/minihack](https://github.com/facebookresearch/minihack)
  - OpenSim
    - [UtkarshMishra04/bioimitation-gym](https://github.com/UtkarshMishra04/bioimitation-gym)
    - [stanfordnmbl/osim-rl](https://github.com/stanfordnmbl/osim-rl)
  - Electric Engineering
    - [upb-lea/gym-electric-motor](https://github.com/upb-lea/gym-electric-motor)
    - [upb-lea/openmodelica-microgrid-gym](https://github.com/upb-lea/openmodelica-microgrid-gym)
    - [tobirohrer/building-energy-storage-simulation](https://github.com/tobirohrer/building-energy-storage-simulation)
    - [intelligent-environments-lab/CityLearn](https://github.com/intelligent-environments-lab/CityLearn) (multiagent)
  - [koulanurag/ma-gym](https://github.com/koulanurag/ma-gym) (multiagent)
  - Multi-armed Bandits:
    - [magni84/gym_bandits](https://github.com/magni84/gym_bandits)
    - [ThomasLecat/gym-bandit-environments](https://github.com/ThomasLecat/gym-bandit-environments)
    - [JKCooper2/gym-bandits](https://github.com/JKCooper2/gym-bandits)
    - [diegoalejogm/openai-k-armed-bandits](https://github.com/diegoalejogm/openai-k-armed-bandits)
  - Another awesome list: [Phylliade/awesome-openai-gym-environments](https://github.com/Phylliade/awesome-openai-gym-environments)

- Unity ML-Agents
  - [Unity-Technologies/marathon-envs](https://github.com/Unity-Technologies/marathon-envs)

Multi-agent:

- [Farama-Foundation/PettingZoo](https://github.com/Farama-Foundation/PettingZoo) (<https://pettingzoo.farama.org>) [![GitHub stars](https://img.shields.io/github/stars/Farama-Foundation/PettingZoo)](https://github.com/Farama-Foundation/PettingZoo/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/Farama-Foundation/PettingZoo?label=last%20update)
  - [Farama-Foundation/MAgent2](https://github.com/Farama-Foundation/MAgent2)
  - [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents) (Environments, Algorithms) (includes design of environments)
  - [LucasAlegre/sumo-rl](https://github.com/LucasAlegre/sumo-rl)

### RL/DRL Benchmarking

With datasets for offline reinforcement learning:

- [Farama-Foundation/D4RL](https://github.com/Farama-Foundation/D4RL) (was [rail-berkeley/d4rl](https://github.com/rail-berkeley/d4rl)) [![GitHub stars](https://img.shields.io/github/stars/Farama-Foundation/D4RL)](https://github.com/Farama-Foundation/D4RL/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/Farama-Foundation/D4RL?label=last%20update)
- [google-research/rlds](https://github.com/google-research/rlds) [![GitHub stars](https://img.shields.io/github/stars/google-research/rlds)](https://github.com/google-research/rlds/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/google-research/rlds?label=last%20update)
- [Farama-Foundation/Minari](https://github.com/Farama-Foundation/Minari) (was [Farama-Foundation/Kabuki](https://github.com/Farama-Foundation/Kabuki)) [![GitHub stars](https://img.shields.io/github/stars/Farama-Foundation/Minari)](https://github.com/Farama-Foundation/Minari/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/Farama-Foundation/Minari?label=last%20update)

With low-cost robots:

- [google-research/robel](https://github.com/google-research/robel) (<https://sites.google.com/view/roboticsbenchmarks>) [![GitHub stars](https://img.shields.io/github/stars/google-research/robel)](https://github.com/google-research/robel/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/google-research/robel?label=last%20update)

Reproducible:

- [rlworkgroup/garage](https://github.com/rlworkgroup/garage) "A toolkit for reproducible reinforcement learning research." [![GitHub stars](https://img.shields.io/github/stars/rlworkgroup/garage)](https://github.com/rlworkgroup/garage/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/rlworkgroup/garage?label=last%20update)

Metrics/benchmarks:

- [stepjam/RLBench](https://github.com/stepjam/RLBench) (<https://sites.google.com/view/rlbench>) "A large-scale benchmark and learning environment." [![GitHub stars](https://img.shields.io/github/stars/stepjam/RLBench)](https://github.com/stepjam/RLBench/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/stepjam/RLBench?label=last%20update)
- [google-research/rliable](https://github.com/google-research/rliable) (<https://agarwl.github.io/rliable>) "reliable evaluation on RL and ML benchmarks, even with only a handful of seeds" [![GitHub stars](https://img.shields.io/github/stars/google-research/rliable)](https://github.com/google-research/rliable/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/google-research/rliable?label=last%20update)
- [google-research/rl-reliability-metrics](https://github.com/google-research/rl-reliability-metrics) "provides a set of metrics for measuring the reliability of reinforcement learning (RL) algorithm" [![GitHub stars](https://img.shields.io/github/stars/google-research/rl-reliability-metrics)](https://github.com/google-research/rl-reliability-metrics/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/google-research/rl-reliability-metrics?label=last%20update)
- [HYDesmondLiu/B2RL](https://github.com/HYDesmondLiu/B2RL) "Building Batch Reinforcement Learning Dataset" [![GitHub stars](https://img.shields.io/github/stars/HYDesmondLiu/B2RL)](https://github.com/HYDesmondLiu/B2RL/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/HYDesmondLiu/B2RL?label=last%20update)
- <https://martin-thoma.com/sota/#reinforcment-learning>

Other:

- [deepmind/bsuite](https://github.com/deepmind/bsuite) "collection of carefully-designed experiments that investigate core capabilities of a reinforcement learning (RL) agent" [![GitHub stars](https://img.shields.io/github/stars/deepmind/bsuite)](https://github.com/deepmind/bsuite/stargazers) ![GitHub last commit](https://img.shields.io/github/last-commit/deepmind/bsuite?label=last%20update)

### RL/DRL Books

RL

- Reinforcement Learning: An Introduction: <http://incompleteideas.net/book/RLbook2020.pdf> (Richard S. Sutton is father of RL)
- Andrew Ng thesis: <www.cs.ubc.ca/~nando/550-2006/handouts/andrew-ng.pdf>
- <https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym>

## Evolutionary Algorithms (EA)

Only accounting those with same objective as RL.

- <https://blog.openai.com/evolution-strategies>
- <https://eng.uber.com/deep-neuroevolution>
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
  - [CMA-ES/pycma](https://github.com/CMA-ES/pycma)
  - [hardmaru/estool](https://github.com/hardmaru/estool)
  - [CyberAgent/cmaes](https://github.com/CyberAgent/cmaes)

## Misc Tools

- DLPaper2Code: Auto-generation of Code from Deep Learning Research Papers: <https://arxiv.org/abs/1711.03543>
- Tip: you can download the raw source of any arxiv paper. Click on the "Other formats" link, then click "Download source"
- <http://www.arxiv-sanity.com>

## Similar pages

- [tigerneil/awesome-deep-rl](https://github.com/tigerneil/awesome-deep-rl)
- [kengz/awesome-deep-rl](https://github.com/kengz/awesome-deep-rl)
- [williamd4112/awesome-deep-reinforcement-learning](https://github.com/williamd4112/awesome-deep-reinforcement-learning)
- [terryum/awesome-deep-learning-papers#new-papers](https://github.com/terryum/awesome-deep-learning-papers#new-papers)
- [hanjuku-kaso/awesome-offline-rl](https://github.com/hanjuku-kaso/awesome-offline-rl)
- [wwxFromTju/awesome-reinforcement-learning-lib](https://github.com/wwxFromTju/awesome-reinforcement-learning-lib)
