from typing import List
import torch
import pandas as pd
import numpy as np
import re
import string
import nltk.data
import nltk
from matplotlib import pyplot as plt
# %matplotlib inline
# Download English
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

pd.set_option('max_colwidth', None)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup


def roberta_preprocess(data):
    # renaming the columns
    data = data.rename(columns={0: 'Sentiment', 1: 'Category', 2: 'Subject', 3: 'Index', 4: 'Text'})

    # Lowercase
    data["Text"] = data["Text"].str.lower()
    data["Subject"] = data["Subject"].str.lower()

    # Special Token
    data[['Start', 'End']] = data['Index'].str.split(':', 1, expand=True)
    data['Start'] = data['Start'].astype(int)
    data['End'] = data['End'].astype(int)
    data['Text'] = data.apply(
        lambda x: x['Text'][:x['Start']] + '\"' + x['Text'][x['Start']:x['End']] + '\"' + x['Text'][x['End']:],
        axis=1)

    data = data.drop(columns=['Index', 'Start', 'End'])

    # Separating 2 columns
    data['Category'] = data['Category'].str.lower()
    data[['Main_Category', 'Sub_Category']] = data['Category'].str.split('#', 1, expand=True)
    data['Sub_Category'] = data['Sub_Category'].str.replace('_', ' ')
    data['Category'] = data['Main_Category'] + ' ' + data['Sub_Category']

    data['Text'] = data['Text'] + ' <s> ' + data['Category'] + ' </s>'
    # data = data.drop(columns='Category')

    # Label Encoding Sentiment, Category, and Sub-Category
    data['Sentiment'] = data['Sentiment'].apply(
        lambda x: 2 if x == 'positive' else (1 if x == 'neutral' else 0))
    # data['Main_Category_Label'] = data['Main_Category'].apply(lambda x: 1 if x == 'AMBIENCE' else (2 if x == 'FOOD' else (3 if x == 'SERVICE' else (4 if x == 'RESTAURANT' else (5 if x == 'DRINKS' else 6)))))
    # data['Sub_Category_Label'] = data['Sub_Category'].apply(lambda x: 1 if x == 'GENERAL' else (2 if x == 'QUALITY' else (3 if x == 'STYLE_OPTIONS' else (4 if x == 'MISCELLANEOUS' else 5))))

    return data

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """

    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

        # Prepare Data
        train = pd.read_csv(train_filename, sep='	', header=None)
        train = roberta_preprocess(train)
        # test = pd.read_csv(dev_filename, sep='	', header=None)
        # test = roberta_preprocess(test)

        # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        train['Input'] = train['Text'].apply(lambda x: tokenizer(x, padding='max_length', max_length=100)['input_ids'])
        train['Mask'] = train['Text'].apply(
            lambda x: tokenizer(x, padding='max_length', max_length=100)['attention_mask'])

        # test['Input'] = test['Text'].apply(lambda x: tokenizer(x, padding='max_length', max_length=100)['input_ids'])
        # test['Mask'] = test['Text'].apply(
        #     lambda x: tokenizer(x, padding='max_length', max_length=100)['attention_mask'])

        X = train[['Input', 'Mask']]
        y = np.array(train['Sentiment'].tolist())
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # X_test = test[['Input', 'Mask']]
        # y_test = np.array(test['Sentiment'].tolist())
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(np.array(X_train['Input'].tolist()), dtype=torch.long),
            torch.tensor(np.array(X_train['Mask'].tolist()), dtype=torch.long),
            torch.tensor(y_train, dtype=torch.long))

        val_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(X_val['Input'].tolist()), dtype=torch.long),
                                                     torch.tensor(np.array(X_val['Mask'].tolist()), dtype=torch.long),
                                                     torch.tensor(y_val, dtype=torch.long))

        # test_dataset = torch.utils.data.TensorDataset(
        #     torch.tensor(np.array(X_test['Input'].tolist()), dtype=torch.long),
        #     torch.tensor(np.array(X_test['Mask'].tolist()), dtype=torch.long),
        #     torch.tensor(y_test, dtype=torch.long))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

        # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

        roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)
        '''
        for param in Roberta.parameters():
            param.requires_grad = False
        '''

        class Model(nn.Module):

            def __init__(self, roberta, device_name):
                super().__init__()

                self.roberta = roberta.to(device_name)
                self.l1 = nn.Linear(in_features=768, out_features=768)
                # self.relu = nn.ReLU(inplace=True)
                self.drop = nn.Dropout(p=0.1)
                self.l2 = nn.Linear(in_features=768, out_features=3)

            def forward(self, x, attention_mask):
                x = self.roberta(x, attention_mask)
                x = x.pooler_output
                x = self.l1(x)
                # x = self.relu(x)
                x = self.drop(x)
                x = self.l2(x)

                return x

        self.model = Model(roberta_model, device).to(device)

        def exec_train(model, loss_fcn, device_name, optimizer, max_epochs, dataloader_train, dataloader_val):

            epoch_list = []
            scores_list = []
            lowest_loss = 1

            # loop over epochs
            for epoch in range(max_epochs):
                model.train()
                losses = []
                # loop over batches
                for i, data in enumerate(dataloader_train):
                    optimizer.zero_grad()
                    inputs, mask, labels = data
                    outputs = model(inputs.to(device_name), mask.to(device_name))
                    # compute the loss
                    loss = loss_fcn(outputs, labels.to(device_name))
                    # optimizer step
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                loss_data = np.array(losses).mean()
                print("Epoch {:05d} | Loss: {:.4f}".format(epoch, loss_data))

                if epoch % 5 == 0:
                    # evaluate the model on the validation set
                    # computes the f1-score
                    score_list_batch = []

                    model.eval()
                    with torch.no_grad():
                        for i, batch in enumerate(dataloader_val):
                            inputs, mask, labels = batch
                            output = model(inputs.to(device_name), mask.to(device_name))
                            loss_test = loss_fcn(output, labels.to(device_name))
                            predict = torch.argmax(output, axis=1)
                            score = accuracy_score(labels.cpu().numpy(), predict.cpu().numpy())
                            score_list_batch.append(score)

                    score = np.array(score_list_batch).mean()
                    print("Accuracy-Score: {:.4f}".format(score))
                    scores_list.append(score)
                    epoch_list.append(epoch)

            return epoch_list, scores_list

        ### Max number of epochs
        max_epochs = 41

        ### DEFINE LOSS FUNCTION
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(device)

        self.loss_fcn = nn.CrossEntropyLoss(weight=class_weights)

        ### DEFINE OPTIMIZER
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, max_epochs * len(train_dataloader))
        _, _ = exec_train(self.model, self.loss_fcn, device, optimizer, max_epochs, train_dataloader,
                                               val_dataloader)

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        data = pd.read_csv(data_filename, sep='	', header=None)
        data = roberta_preprocess(data)
        # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        data['Input'] = data['Text'].apply(lambda x: tokenizer(x, padding='max_length', max_length=100)['input_ids'])
        data['Mask'] = data['Text'].apply(
            lambda x: tokenizer(x, padding='max_length', max_length=100)['attention_mask'])

        X = data[['Input', 'Mask']]
        # y = np.array(data['Sentiment'].tolist())

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(np.array(X['Input'].tolist()), dtype=torch.long),
            torch.tensor(np.array(X['Mask'].tolist()), dtype=torch.long))
            # torch.tensor(y, dtype=torch.long))

        # inputs = torch.tensor(np.array(X['Input'].tolist()), dtype=torch.long)
        # mask = torch.tensor(np.array(X['Mask'].tolist()), dtype=torch.long)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

        self.model = self.model.to(device)
        self.model.eval()
        predict = []
        for i, batch in enumerate(dataloader):
          inputs, mask = batch
          output = self.model(inputs.to(device), mask.to(device))
          predict += torch.argmax(output, axis=1).tolist()

        return ['positive' if x == 2 else 'neutral' if x == 1 else 'negative' for x in predict]



