# Interior Wall & Floor Segmentation

### Whole Project in a glance

I have experimented with 4 approches for this project, where two are from core computer vision and rest 2 are deep learning based.

The problem with only computer vision is that- most of the time it will manual as we are playing with thresholds. And for edge detction, gry scales of the images will reduces the info. While it is efficient and working pretty much good on the simpler problems, where the room is empty and not so much objects are there. When the walls and floor are seprated by the straight line v1 works so well. As it detects throgh the horizontal edges who are greater then the threshold.

for v2 in core computer segment some non-linearity and more spatial intelligances added. But it performs poor. In simple cases it add un-wanted non-lineaty and in non-liner cases it is less capable.

Then it comes to the deep learning based approach, the v1 and v2 raw U-net segmenation models are trained on around 600 images from scratch. V1 encodes on 2d-cnn layers and v2 on resnet(weights = flase) with batch norm and early stopping.V1 performs well on simple sptial images and v2 works fine with more complex and noise conditions. However there are some signs of overfitting because of tiny dataset but patetions parmeter will stopped the training at 65 epochs. All the graphs and inference samples are below.


---

## Project Structure

```text
wall_floor_seg/
│
├── core_cv/
│   ├── v1/
│   └── v2/
│
├── deep_learning/
│   ├── v1/
│   └── v2/
│
├── dataset/
│
├── predictions/
│
└── README.md
```

---

# 1. Core Computer Vision based approches

The first stage was implemented without using a deep learning model. It has two versions - v1 and v2


---

## V1 — Geometric Approach

### assumption :

**wall and floor** can generally be represented by a horizontal line which is large most of the cases.

The pipeline was approximately:

```text
Input Image
     ↓
Preprocessing
     ↓
Canny Edge Detection
     ↓
Morphology
    ↓
Lines Detection & hight estimation
     ↓
Horizontal Line Filtering
     ↓
Wall / Floor Boundary
     ↓
Segmentation
```
<table>
  <tr>
    <td><img src="assets\images.jpg" width="300"></td>
    <td><img src="assets\output.png" width="300"></td>
    <td><img src="assets\image.png" width="300"></td>
    <td><img src="assets\op.png" width="300"></td>
  </tr>
</table>


### Limitation

It only creates liner boundries. Performe well on simple edge cases. But when it comes to a bit tougher non liner boundry. it failed.

<table>
  <tr>
    <td><img src="assets\images.jpg" width="300"></td>
    <td><img src="assets\output.png" width="300"></td>
    <td><img src="assets\image.png" width="300"></td>
    <td><img src="assets\op.png" width="300"></td>
  </tr>
</table>

---

# V2 — Non-linearity + Spatiality



### Improvements

* Color differences 
* Pixel intensity variation
* Spatial regions
* Non-linear wall/floor boundaries

General pipeline:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Color / Intensity Analysis
     ↓
Region Segmentation
     ↓
Boundary Refinement
     ↓
Wall / Floor Mask
```

This approach produced significantly better segmentation compared with the first classical version, particularly in cases where the wall-floor boundary was not a clean horizontal line. But it also didn't seems perfact.

### Limitations
However we are able to make the non-linear mask and some spatal understanding but that is still not much precise as the whole pipline
is based on the thresholing and manual intervensations. so i decided to move toward deep-learning. While remberiring that im not 
adviced to used any pre-train model or services. Here are few example of this v2 approach :

<table>
  <tr>
    <!-- <td><img src="assets\download.png" width="300"></td> -->
    <td><img src="assets\download1.png" width="300"></td>
    <td><img src="assets\download2.png" width="300"></td>
  </tr>
</table>

---


# 2. Deep Learning based approches

After didn't getting the desired results. I Decded to move into the u-net model while keeping the constraints in mind. Training v1 and v2 u-net model on custome archetecture
from use scratches with out any pre-defined weigts. Both model will  train only on 700 img dataset with collab T4 GPU. Here is the config of dataset- 

<img src="assets\dataset.png">

---

## V1 — U-Net with Custom CNN Encoder

The first deep learning version uses a standard U-Net style architecture with a custom CNN encoder.

### Parameters
* Custom CNN encoder
* Loss - Cross Entropy Loss
* Optimizer - ADAM
* Convolution + Batch Normalization + ReLU blocks
* SEED = 42
* NUM_CLASSES = 3
* IMG_W, IMG_H = 512, 288
* BATCH_SIZE = 16
* EPOCHS = 100
* LR = 3e-4
* WEIGHT_DECAY = 1e-4
* PATIENCE = 12


### Evals on Test Dataset


<img src="assets\v1_test_eval.png">

### Inference Samples


Majorly connected and big spatial structurs readed well by model. Works well for the raw rooms where there are not many objects. In the noisy structures it
struggles.

<img src="assets\v1_inference_sample.png">

<img src="assets\v1_inference_sample2.png">

<img src="assets\v1_pred3.png">




---

# V2 — U-Net with ResNet Encoder

The custom CNN encoder is relatively simple and learns all visual features from the dataset. Using Resnet deeper feture extraction is possible as we trained from sctrach so it required for quite better performance.


### Architecture

```text
* Encoder - ResNet(weights = False)
* Loss - Cross Entropy Loss
* SEED = 42
* NUM_CLASSES = 2
* IMG_W = 512
* IMG_H = 288
* BATCH_SIZE = 16
* EPOCHS = 100
* LR = 3e-4
* WEIGHT_DECAY = 1e-4
* PATIENCE = 15
```

### Eval on Test data

<img src="assets\v2_test.png">


### Training curvs


<table>
  <tr>
    <td><img src="assets\tarinvsval.png" width="300"></td>
    <td><img src="assets\trainvsvaldice.png" width="300"></td>
  </tr>
    <tr>
    <td><img src="assets\trainvsvaliou.png" width="300"></td>
    <td><img src="assets\lrrate.png" width="300"></td>
  </tr>
</table>

some overfitting is there due to the very tiny data set but training will stopped on 65 epoches by early stopping param.

## Infernces 

<img src="assets\v1v2_pred.png">
<img src="assets\v1v2_pred1.png">
<img src="assets\v1v2_pred3.png">


---

# comparison in V1 and V2

<img src="assets\mega2.png">
<img src="assets\mega_pred.png">
<img src="assets\mega5.png">




---

