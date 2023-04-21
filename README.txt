{\rtf1\ansi\ansicpg1252\cocoartf2706
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica-Bold;\f1\fnil\fcharset0 HelveticaNeue-Bold;\f2\fswiss\fcharset0 Helvetica;
\f3\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid1\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid101\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid2}
{\list\listtemplateid3\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid201\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid3}
{\list\listtemplateid4\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid301\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid4}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}{\listoverride\listid3\listoverridecount0\ls3}{\listoverride\listid4\listoverridecount0\ls4}}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\qc\partightenfactor0

\f0\b\fs36 \cf0 NLP Opinio
\f1 n
\f0  Polarity 
\f1\fs26 \

\f2\b0\fs24 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \

\f0\b\fs28 Studens 
\f2\b0  
\fs24 \
Darius CHUA\
Karim EL HAGE\
Deepesh DWIVEDI
\f0\b \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f2\b0 \cf0 Yasmina HOBEIKA\

\f0\b \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\fs28 \cf0 Classifier 
\f2\b0\fs24 \
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b\fs26 \cf0 1. Introduction:
\f3\b0 \
This sentiment analysis classifier is designed to predict the sentiment of a given text as either 'positive', 'neutral', or 'negative'. The model leverages the pre-trained RoBERTa (Robustly optimized BERT approach) model, which is a variation of the popular BERT model. RoBERTa is an advanced language understanding model known for its strong performance on a wide range of natural language processing tasks.\
\

\f1\b Model Architecture:
\f3\b0 \
The model architecture consists of the following components:\
\pard\pardeftab560\pardirnatural\partightenfactor0
\ls1\ilvl0\cf0 {\listtext	1.	}RoBERTa Model: The pre-trained RoBERTa model (base version) is used as a feature extractor. It provides contextualized word embeddings, which are essential for capturing the semantics of text data. The model is fine-tuned during training to adapt to the specific sentiment analysis task.\
{\listtext	2.	}Linear Layers: Two linear layers are used to map the output of the RoBERTa model to the final sentiment class probabilities. The first layer has 768 input features and 768 output features, followed by a dropout layer with a dropout rate of 0.1 for regularization. The second linear layer maps the 768 features to the 3 output sentiment classes ('positive', 'neutral', and 'negative').\
\pard\tx720\pardeftab560\pardirnatural\partightenfactor0
\cf0 \
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b \cf0 2. Preprocessing:
\f3\b0 \
The input data undergoes several preprocessing steps before being fed into the model:\
\pard\pardeftab560\pardirnatural\partightenfactor0
\ls2\ilvl0\cf0 {\listtext	1.	}Renaming columns: The columns in the input data are renamed for easier understanding and manipulation.\
{\listtext	2.	}Lowercasing: Both the 'Text' and 'Subject' columns are lowercased to reduce the impact of case differences on the model.\
{\listtext	3.	}Special token insertion: Special tokens are inserted into the text to help the model understand the structure of the input.\
{\listtext	4.	}Column separation: The 'Category' column is separated into 'Main_Category' and 'Sub_Category' columns for better context representation.\
{\listtext	5.	}Label encoding: Sentiment labels are numerically encoded for model compatibility (positive: 2, neutral: 1, negative: 0).\
\pard\tx720\pardeftab560\pardirnatural\partightenfactor0
\cf0 \
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b \cf0 3. Training:
\f3\b0 \
The model is trained using the following procedure:\
\pard\pardeftab560\pardirnatural\partightenfactor0
\ls3\ilvl0\cf0 {\listtext	1.	}Data preparation: The input text data is tokenized using the RoBERTa tokenizer, with padding and truncation applied to create input sequences of fixed length (100 tokens).\
{\listtext	2.	}Dataset creation: Training and validation datasets are created using a 80-20 train-validation split, with data loaders configured for efficient batch processing.\
{\listtext	3.	}Loss function: Cross-entropy loss with class weights is used to handle imbalanced class distributions in the training data.\
{\listtext	4.	}Optimizer: The AdamW optimizer with a learning rate of 2e-5 is employed for model training.\
{\listtext	5.	}Scheduler: A linear learning rate scheduler with warmup is utilized to optimize the learning rate during training.\
{\listtext	6.	}Training loop: The model is trained for a total of 21 epochs. Performance is evaluated on the validation set every 5 epochs using the accuracy score as the evaluation metric.\
\pard\tx720\pardeftab560\pardirnatural\partightenfactor0
\cf0 \
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b \cf0 4. Prediction:
\f3\b0 \
The classifier's 'predict' method takes an input data filename and a device as arguments. The input data is preprocessed, tokenized, and loaded into a data loader. The trained model is then used to generate sentiment predictions for the input data, which are converted back to the original label format ('positive', 'neutral', 'negative') and returned as a list of predictions.\
\pard\tx720\pardeftab560\pardirnatural\partightenfactor0
\cf0 \

\f1\b 5. Accuracy:\

\f3\b0 To calculate the accuracy we are using F1 Score since the training data set is not balanced hence we chose F1 score.\

\f1\b \
\pard\tx220\tx720\pardeftab560\li720\fi-720\pardirnatural\partightenfactor0
\ls4\ilvl0
\f3\b0 \cf0 {\listtext	1.	}Train Set - \
{\listtext	2.	}Dev Set - \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f2\fs24 \cf0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\qc\partightenfactor0
\cf0 \
}