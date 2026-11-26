
```text
PPE Compliance for Factory Workers

Yolov11 , RTSP, Real-time video Analytics, inference optimization with TensorRT,
ONNX at int16 at 15fps, with deepstream. custom dataset.
ByteTrack for MOT, temporal voating with 10 continoues frames.
```

```text
3D Geometrical re-construction with AR/VR data.

Optimized jacobian odometry, SLAM , 3d point-clouds and Mesh, TSDF fusion.
camera intrinsics and extrinsics and pose estimation, depth maps.
```

```text
Stable diffsuion + LoRA for text-to-image
unet+vae and CLIP for text encoding and contrastive loss.
```

```text
Sageformer-B01  for image segmentation

Training a self attetention based MLP(multi-layer-perceptron) transformer model
for semantic segmentation of liquid spills. Creating an custom dataset for the
indoor and outdoor conditions as well as make sure it perfomed well at tiny,
meduim, large and transparent spills as water as well. Adding hard negative and
prevent data lekage by p-hashing clustering, to be reducing the number of
Flase-Negatives which helps to achieve 91% precsions in such a false-alarming-prone
problem. Used cross entroplpy and dice loss with the weitage of 50-50 to be
acheved precisise binary-class calassification and tiny object detection.
```

```text
U-net with ResNet Encoder

U-net with Reset-50 encoder for the obj wall floor boundries, used small ~1.5k image dataset
labeled with automated pipeline of SAM and YOLO-world with human optimization and with
horizontal flipping augumentations, HE weight initlization and batchNorm layers for normal
data distributions will used resudual learning blocks which is reducing the risk of vanishing-gradient
and softmax activation function in the fully connected layers for getting the probability distributions
amoung the classes. Got 90 mIOU score with multiclass classification, Mlflow and Tensorboard
used for model and dataset ragistry and tracking training curvs.
```

```text
Real-time obj tracking algorthem(No deep learning)

Performing linear algebera based 2d tensor computer vision algorithms to detect the cars
running on the roads from a stable camera setup. Performing the algorithem object detection
by the background substractor on edges extracted by the canny edge detector. Performing
gaussian filetering and moropholy for smothenout the output. Detections are based on hand-crafted
fetures based on the static thresholds based temporal voating.
Backgroud substractor, Morophology, Temporal voating, template matching

```

```text
PaddleOCR det and rec for OCR

Fine-tuning the paddle two models - PaddleOCR v4 det and PaddleOCR v4 recognition,
camparing the results from previous v3 versions, do hyperparameter twiking, optimized the
config.yml file with cosine learning rate scheduling and L2 regulirazation for reducing overfitting risk.
Used paddlepaddle-GPU hardware accelaration. Rec is build on CNN backbone with the global/local mixing
blocks for efficient lower level device support. This whole model is trained on Cloud Nvidia T4 GPU with
12 GB of vRAM.
```


```text
Fine-tuning speaker dairization Pyannote

Fine-tuning a speaker dairization model from pyannote using huggingFace librbary,
improving the voice-activation-detection(VAD) models used in voice-bots improved
the resopisivness and over all user experience with more realistic way to intruppt
the conversasations. Performing the evaluation of fine-tuned model with test dataset
and got 8% reduction in DER(Dairization error rate) from the base model.
Voice activity detection, hugging face , librosa, DER, male spectogram
```

```text
[CNN based audio-classifier for voice-male detection]

Build an end-to-end CNN training pipeline with 3 conv2d layers and adaptive average pooling
for identifying and etracting   the patterns from audio data, converting wav files into
male-spectogram for image like signal processing. collected custom voice-male sound samples,
beeps and train a voice-male detector or classifer for voice bots. To save the lot of API
costing in TTS-LLM-TTS based voice-bots used in cold-calling.
audio CNN model, Male spectogram, Signal Processing
```

```text
Webrtc based voice-agent

webrtc, tts, stt, llm, tcp/udp, orcherestreation, RAG , Prompt Engineering , optimization, GCP, Docker
```

```text
Node orchresation based CS agent

hybrid search(BM25 + TS-IDF) , Web Search implemented, postgres sql, Re-ranking RAG
```


