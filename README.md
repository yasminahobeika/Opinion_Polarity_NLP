# Natural Language Processing Assignment - Opinion Polarity Sentiment Analysis Task

## Authors
The project is done by [Darius CHUA](https://github.com/darius-chua), [Karim EL HAGE](https://github.com/karimelhage), [Yasmina HOBEIKA](https://github.com/yasminahobeika), [Deepesh DWIVEDI](https://github.com/deepesh-dwivedi)

## Results
The average accuracy on the dev set provided after conducting 5 runs from scratch:

- Accuracy: **87.93%**
- Exec Time: **2048.87s (409 per run)**

## Classifer:
### I. Introduction

This sentiment analysis classifier is designed to predict the sentiment of a given text as either 'positive', 'neutral', or 'negative'. The model leverages the pre-trained RoBERTa (Robustly optimized BERT approach) model, which is a variation of the popular BERT model. RoBERTa is an advanced language understanding model known for its strong performance on a wide range of natural language processing tasks.

### II. Pre-Processing

The input data undergoes several preprocessing steps before being fed into the model:

1. **Renaming columns:** The columns in the input data are renamed for easier understanding and manipulation.
2. **Lowercasing:** Both the 'Text' and 'Subject' columns are lowercased to reduce the impact of case differences on the model.
3. **Special token insertion:** Special tokens are inserted into the text to help the model understand the structure of the input. Namely for the location of the part of the Text that cause the sentiment as well as its aspect category. 
4. **Label Encoding:** Sentiment labels are numerically encoded for model compatibility (positive: 2, neutral: 1, negative: 0).

### III. Model Architecture

The model architecture consists of the following components:

1. **RoBERTa Model:** The pre-trained RoBERTa model (base version) is used as a feature extractor. It provides contextualized word embeddings, which are essential for capturing the semantics of text data. The model is fine-tuned during training to adapt to the specific sentiment analysis task.
2. **Linear Layers:** Two linear layers are used to map the output of the RoBERTa model to the final sentiment class probabilities. The first layer has 768 input features and 768 output features, followed by a dropout layer with a dropout rate of 0.1 for regularization. The second linear layer maps the 768 features to the 3 output sentiment classes ('positive', 'neutral', and 'negative').

### IV. Training

The model is trained using the following procedure:

1. **Data preparation:** The input text data is tokenized using the RoBERTa tokenizer, with padding and truncation applied to create input sequences of fixed length (100 tokens).
2. **Dataset creation:** Training and validation datasets are created using a 80-20 train-validation split, with data loaders configured for efficient batch processing.
3. **Loss function:** Cross-entropy loss with class weights is used to handle imbalanced class distributions in the training data.
4. **Optimizer:** The AdamW optimizer with a learning rate of 2e-5 is employed for model training.
5. **Scheduler:** A linear learning rate scheduler with warmup is utilized to optimize the learning rate during training.
6. **Training loop:** The model is trained for a total of 21 epochs. Performance is evaluated on the validation set every 5 epochs using the accuracy score as the evaluation metric.
