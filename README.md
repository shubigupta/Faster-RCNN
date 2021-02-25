# Faster RCNN: A two-stage object detector

Faster RCNN is two-stage RCNN based object classifier. The first stage involves the use of Region Proposal Network(RPN) which is run once per image. It is responsible for:
1. classifying whether an anchor box is object or not
2. predicting the transform from anchor box to proposal box.

The second stage runs once per region proposed by RPN. It involves:
1. Cropping features: ROI pool/align
2. Classifying proposals as background or object
3. Predicting transform from proposal box to object box

For this project, a subset of COCO dataset was used to detect 3 types of objects: Vehicles, People, and Animals.

## Part A: Training RPN
No. of Anchors used: 1
<table>
  <tr>
      <td align = "center"> <img src="./Results/1. RPN backbone.png"> </td>
  </tr>
  <tr>
      <td align = "center"> Backbone used for RPN</td>
  </tr>
</table>

**Intermediate** **Layer:**  Kernel Size: (3,3,256), Batch Norm, ReLU, Padding = Same

**RPN Classifier Head:** Kernel Size: (1,1,1), Sigmoid, Padding = Same

**RPN Regressor Head:**  Kernel Size: (1,1,4), Padding = Same

### RPN Losses:
**Classifier** **Loss:** Binary Cross Entropy.
**Regressor** Loss:**  Smooth L1 Loss. **Note:** To minimize the bias towards negative classes, the positive(object) and negative(not object) anchors are subsampled in a ratio as close as possible to 1:1. Also, only a constant size minibatch is used for training regressor.

### Post Processing:
Boxes going out of the image boundary are removed. After that top K boxes(top objectness score boxes) are kept which is followed by Non-Maximum Suppression. The top N boxes after NMS are then kept as final proposals.

### Results:
<table>
  <tr>
      <td align = "center"> <img src="./Results/2. Pre NMS 1.png"> </td>
      <td align = "center"> <img src="./Results/3. Post NMS 1.png"> </td>
  </tr>
  <tr>
      <td align = "center"> <img src="./Results/4. Pre NMS 2 .png"> </td>
      <td align = "center"> <img src="./Results/5. Post NMS 2.png"> </td>
  </tr>
  <tr>
      <td align = "center"> Pre NMS, K = 20</td>
      <td align = "center"> POST NMS, N = 5s </td>
  </tr>
</table>

<table>
  <tr>
      <td align = "center"> <img src="./Results/6. Confidence error.png"> </td>
      <td align = "center"> <img src="./Results/7. Localisation and classification error.png"> </td>
  </tr>
  <tr>
      <td align = "center"> Confidence error </td>
      <td align = "center"> Localisation and Classification error </td>
  </tr>
</table>
### Loss curves:
<table>
  <tr>
      <td align = "center"> <img src="./Results/6. Classifier Loss.png"> </td>
      <td align = "center"> <img src="./Results/7. Regressor Loss.png"> </td>
  </tr>
  <tr>
      <td align = "center"> Classifier Loss Curve</td>
      <td align = "center"> Regressor Loss Curve</td>
  </tr>
</table>
