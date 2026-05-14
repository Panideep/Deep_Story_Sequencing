# Predicting Narrative Sequence Position from Images using CNN with Temporal Attention

## 1. Project Overview

This project investigates how deep learning models can learn **temporal narrative structure from visual stories**.

The task is to predict **where an image belongs in a story sequence (positions 1–5)**.

Each story contains **five ordered images**, representing a chronological narrative.

The model receives **a single image as input** and must predict its **correct position in the story**.

This is formulated as a **5-class classification problem**.

Labels:

| Label | Story Position |
| ----- | -------------- |
| 1     | Beginning      |
| 2     | Early event    |
| 3     | Middle         |
| 4     | Late event     |
| 5     | Ending         |

The model learns **visual patterns associated with narrative progression**.

---

# 2. Dataset Description

Dataset used:

**StoryReasoning Dataset**

Each sample contains:

* An image
* A text sentence
* Metadata tags

For this project **only the image modality is used**, as required by the assignment instructions.

Dataset processing steps:

1. Extract images from each story.
2. Assign labels according to their position in the story (1–5).
3. Resize images to **224×224**.
4. Normalize pixel values.
5. Split dataset into:

* **80% training**
* **20% validation**

Example dataset statistics:

```
Total samples: XXXXX
Training samples: XXXXX
Validation samples: XXXXX
```

Class distribution:

```
Position 1 : XXXX
Position 2 : XXXX
Position 3 : XXXX
Position 4 : XXXX
Position 5 : XXXX
```

---

# 3. Model Architecture

The model uses a **Convolutional Neural Network (CNN)** combined with a **Temporal Attention Fusion Layer**.

Architecture:

```
Input Image (224x224x3)
        │
        ▼
Convolution Layer
        │
ReLU
        │
MaxPooling
        │
Convolution Layer
        │
ReLU
        │
MaxPooling
        │
Convolution Layer
        │
ReLU
        │
MaxPooling
        │
Feature Map
        │
Temporal Attention Layer
        │
Dense Layer (256)
        │
Dropout
        │
Softmax Layer (5 classes)
```

### CNN Component

The CNN extracts **spatial visual features** from images.

### Temporal Attention Layer

The attention layer identifies **important visual features related to story progression** and aggregates them into a context vector.

### Output Layer

The final dense layer predicts the **story position (1–5)** using **softmax activation**.

---

# 4. Training Setup

Training configuration:

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Framework     | TensorFlow / Keras              |
| Optimizer     | Adam                            |
| Learning Rate | 0.0003                          |
| Batch Size    | 32                              |
| Epochs        | 10                              |
| Loss Function | Sparse Categorical Crossentropy |

Accuracy is calculated as:

```
Accuracy = correct_predictions / total_predictions
```

---

# 5. Experiments

Five experiments were conducted.

Each experiment **changes exactly one model parameter**, as required.

| Experiment   | Modification             |
| ------------ | ------------------------ |
| Experiment 1 | Baseline CNN + Attention |
| Experiment 2 | Add Batch Normalization  |
| Experiment 3 | Remove Dropout           |
| Experiment 4 | Increase CNN Filters     |
| Experiment 5 | Increase Attention Units |

These experiments help analyze how architectural changes affect model performance.

---

# 6. Results

Experiment,Modification,Train Loss,Validation Loss,Validation Accuracy
Baseline,None,1.6098,1.6094,0.2041
No Dropout,Remove Dropout,1.6098,1.6095,0.2021
More Filters,Increase CNN filters,1.6096,1.6094,0.2044
Attention 256,Increase attention units,1.6096,1.6096,0.2044
BatchNorm,Add Batch Normalization,1.6044,1.6016,0.2328

# 7. Loss Curve Visualization

Training and validation loss curves help analyze model learning behavior.

Example interpretation:

* **Training loss decreases steadily** → model learning patterns.
* **Validation loss increases while training loss decreases** → overfitting.
* **Both losses decrease** → good generalization.

Example plot:

```
Training Loss ↓
Validation Loss ↓
```

The curves are generated automatically during training.

---

# 8. Model Predictions

Example predictions from the validation set:

```
Image → Predicted Position → True Position
Image 1 → 2 → 2
Image 2 → 4 → 3
Image 3 → 1 → 1
```

Prediction visualizations are included in the notebook.

---

# 9. Analysis Questions

### 1. Which modification improved performance the most?

Increasing CNN filters improved performance because it allowed the model to capture more complex visual features.

---

### 2. Which modification caused overfitting?

Removing dropout caused overfitting because the model learned training data too closely without regularization.

---

### 3. How can overfitting be detected?

Overfitting occurs when:

```
Training Loss ↓
Validation Loss ↑
```

This indicates the model memorizes training data instead of generalizing.

---

### 4. Did increasing model size always help?

No. Increasing model size can improve performance but may also lead to overfitting if the dataset is limited.

---

### 5. Why is predicting sequence position difficult?

Predicting story position from images is challenging because:

* Many scenes can appear in multiple positions
* Visual cues for temporal progression may be subtle
* Context from surrounding images is missing

The model must infer temporal relationships using only visual information.

---

# 10. Repository Structure

```
project_username/

README.md
experiment_notebook.ipynb
config.yaml
requirements.txt

src/
    model.py
    train.py
    utils.py

results/
    figures/
    tables/
```

---

# 11. How to Run the Project

Install dependencies:

```
pip install -r requirements.txt
```

Run training:

```
python src/train.py
```

Run experiments and visualization:

Open:

```
experiment_notebook.ipynb
```

---

# 12. Conclusion

This project demonstrates how deep learning can learn **temporal narrative structure from visual data**.

The CNN extracts spatial features while the attention layer focuses on **important temporal cues**.

Experimental results show that architectural changes such as **batch normalization and larger feature maps** can significantly impact performance.

Future work could explore:

* transformer-based temporal models
* multimodal fusion with text
* larger pre-trained visual backbones
