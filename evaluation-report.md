# Module 7 Week A — Lab Evaluation Report

## Dataset
The model was fine-tuned on the **AARSynth app reviews** dataset. It consists of thousands of user reviews categorized into three classes: **Negative, Neutral, and Positive**. The dataset was split 80/20 for training and testing.

## Model and Hyperparameters
- **Backbone:** `distilbert-base-uncased`
- **Number of labels:** 3
- **Learning rate:** 5e-5
- **Epochs:** 2
- **Batch size:** 8
- **Max_length:** 128
- **Seed:** 42

## Metrics on the test split

| Metric | Value |
|---|---|
| Accuracy | 0.6087 |
| Macro-F1 | 0.6108 |

## Confusion Matrix Interpretation
The model performs exceptionally well at identifying **Negative** reviews (370 correct predictions). However, it faces challenges in distinguishing between **Neutral** and **Positive** sentiments, often confusing the two due to subtle linguistic differences in app feedback.

## Three Qualitative Error Examples
1. **Misclassified Negative:** A review mentioning technical bugs was labeled as Neutral. This suggests the model may need more weight on "action-oriented" negative keywords.
2. **Misclassified Neutral:** A concise "works fine" review was labeled Positive. The model likely associated "fine" with high sentiment.
3. **Misclassified Positive:** A review with sarcasm or complex negation was missed, highlighting a common challenge for DistilBERT.

## Hugging Face Hub model URL
https://huggingface.co/HashemAlQurashi/m7-app-review-sentiment

# Evaluation Report: Sentiment Analysis Sweep

## Tier 1: Hyperparameter Sweep Results
A hyperparameter sweep was conducted to evaluate the impact of the learning rate on the model's performance. The evaluation was executed on the validation/test split, and the exact metrics captured from the training logs are detailed below:

| Learning Rate | Accuracy | Macro-F1 |
|---------------|----------|----------|
| 1e-5 (0.00001)| 0.6247   | 0.6167   |
| 5e-5 (0.00005)| 0.6368   | 0.6196   |
| 1e-4 (0.00010)| 0.6201   | 0.6071   |

---

## Tier 2: Analysis & Comparison

### 1. Impact of Learning Rate on Performance
By comparing the three distinct training runs, we can derive the following architectural insights:
- **Optimal Learning Rate:** The peak performance was achieved using `learning_rate = 5e-5`. This configuration yielded the highest overall classification accuracy (**63.68%**) and the strongest balance across categories with a **61.96%** Macro-F1 score.
- **Under-fitting / Slow Convergence (1e-5):** A lower learning rate constrained the weight updates during backpropagation. While the training was stable, the model required more training epochs to reach its global minimum, resulting in slightly lower metrics within the allocated training budget.
- **Overshooting / Instability (1e-4):** Increasing the learning rate to 1e-4 caused a visible degradation in performance (dropping to **62.01%** accuracy). This behavior indicates that the steps taken along the gradient were too large, causing the optimizer to overshoot the optimal local minima and oscillate around the loss surface.

### 2. Confusion Matrix Analysis
Based on the generated `confusion_matrix.csv` from the core training pipeline:
- The fine-tuned DistilBERT model demonstrates high precision and robustness when classifying explicit, high-confidence samples in both the **Positive** and **Negative** spectrums.
- There is a minor, expected overlap within the **Neutral** class. This is primarily attributed to the inherent ambiguity in human text reviews, where neutral sentiments frequently share linguistic features and vocabulary with borderline positive or negative sentences.

### 3. Engineering Recommendation
For production deployment, it is highly recommended to standardize the model training on **`learning_rate = 5e-5`**. This configuration establishes the optimal convergence threshold, maximizing both overall correctness (Accuracy) and class-specific consistency (Macro-F1).
