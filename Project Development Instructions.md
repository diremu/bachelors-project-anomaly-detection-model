## Introduction

The purpose of this project is to create an anomaly detection dataset, aimed primarily at Nigerian correctional facilities and the problems they encounter. To create a model capable of achieving the stated objective, follow the instructions below.

### Materials Required

The datasets used for this project are the [UCF-Crime](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset), [Avenue](https://www.kaggle.com/datasets/janeshvarsivakumar/avenue-dataset), and [Shanghai Tech](https://www.kaggle.com/datasets/tthien/shanghaitech).

### Preprocessing

The images are standardized at 224 by 224px. Color conversion is then performed on the images from BGR to RGB. The resulting raw pixel values are then scaled down to a range of 0.0 to 1.0 by dividing by 255 and further normalized using ImageNet mean and standard deviation. Consecutive images are then grouped by 16 into fixed-length frames. A stride of 8 frames is used so behavioural changes between clips are caught. Data augmentation and Dataset Splitting are also performed on the available datasets. 

### Data Loader

A data loader is necessary to reduce the size of the data being inputted at once.The native Python Data Loader is competent enough for this task. The training data loader serves training clips in a normal order, shuffled randomly for each epoch so the model doesnt memorize order. The evaluation data loader serves both normal and anomalous clips, unshuffled, no augmentation. Instead of passing one clip at a time, the data loader passes batches with a tensor shape of (8, 16, 3, 224, 224).

### Model Architecture

For the Model architecture, A CNN processes the spatial features in a frame and an LSTM tracks spatial features over time. ResNet 18 is a good base CNN model to train with due to its lightweightedness, speed and consistent use. The best approach to training with this is partial fine-tuning by freezing the early layers of ResNet and unfreeing the later layers. The LSTM then receives the data from the CNN and should have 2 stacked layers, a hidden size of 512 and unidirectional. A reconstruction based approach would be best to create anomaly scores

### Training Loop

The training loop is the process by which the model is repeatedly shown normal clips, measures how poorly it reconstructs them, and adjusts its weights to improve over time. This continues across multiple epochs until the model has learned normal behaviour well enough that anomalous clips produce a noticeably higher reconstruction error.

Each epoch begins by fetching a batch of 8 normal clips from the training loader. The batch is passed through the full network in a forward pass — ResNet-18 extracts a spatial feature vector from each of the 16 frames per clip, the LSTM processes the resulting sequence of feature vectors to build a temporal representation, and the decoder attempts to reconstruct the original feature sequence from that representation. The reconstruction error between the original and decoded feature vectors is then computed using Mean Squared Error (MSE) as the loss function. A backward pass is then performed, where PyTorch computes the gradient of the loss with respect to every trainable weight in the network. The Adam optimizer, set at a learning rate of 1e-4, then uses those gradients to update the weights in the direction that reduces the loss. This repeats for every batch in the training set until the epoch is complete.

After each epoch, the model switches to evaluation mode and runs through the validation loader without updating any weights. The average reconstruction error is computed separately for normal and anomalous clips. The gap between these two values reflects how well the model has learned to distinguish normal from abnormal behaviour, and AUROC is computed here to track progress. If the validation loss is the lowest recorded so far, the model weights are saved as a checkpoint. If the validation loss shows no improvement for 10 consecutive epochs, training halts early to prevent overfitting and wasted computation. An optional learning rate scheduler halves the learning rate if the validation loss plateaus for 5 epochs, allowing finer weight adjustments in the later stages of training.

### Anomaly Scoring

Once training is complete, the model is used to assign an anomaly score to every clip in the validation and test sets. Each clip is passed through the trained network in the same way as during training — through the CNN backbone, LSTM, and decoder — but no weight updates occur. The reconstruction error, computed as the MSE between the original feature sequence and the reconstructed output, serves as the anomaly score for that clip. A low score indicates the clip closely resembles normal behaviour the model has already learned. A high score indicates the clip deviates significantly from learned normal patterns and is therefore likely anomalous.

All clips are then ranked in descending order of their anomaly scores. No fixed threshold is applied, meaning the system does not make binary decisions about whether a clip is anomalous or not. Instead, the highest-scoring clips are surfaced for human review, with analysts able to inspect the video segments visually and confirm whether the flagged behavior is genuinely anomalous. A temporal consistency check is also applied, where a clip is only considered a strong candidate for anomaly if several consecutive clips around it also score highly, reducing false positives caused by isolated reconstruction noise.

### Evaluation

The model is evaluated on the held-out test set, which contains both normal and anomalous clips that were never seen during training or validation. Performance is measured using three metrics. The primary metric is the Area Under the Receiver Operating Characteristic Curve (AUROC), which measures how well the anomaly scores separate normal clips from anomalous ones across all possible thresholds. A score of 1.0 represents perfect separation and 0.5 represents random guessing. The secondary metrics are Precision, Recall, and F1 score, computed at a range of score thresholds to assess the model's reliability in correctly identifying anomalous clips while minimising false positives. Frame-level AUC is also computed for the UCF-Crime dataset specifically, as this is the standard evaluation protocol used in the literature for that benchmark. Results are compared against similar unsupervised methods from related works to contextualise the model's performance.s