import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

import time
import json

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

df = pd.read_csv('dataset/IMDB/IMDB Dataset.csv', sep=',', encoding='utf-8')
df.head()

# check for missing values
df.isnull().sum()

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Load & Convert contractions dictionary to a more usable format
contraction_df = pd.read_csv('dataset/IMDB/contractions.csv')
contractions_map = dict(zip(contraction_df['Contraction'].str.lower(), contraction_df['Meaning'].str.lower()))

# Define a function for text cleaning
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove emojis
    # Remove non-alphanumeric characters (keeping whitespace and commas for contraction replacement)
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Replace contractions
    text_split = text.split()
    text_split = [contractions_map.get(word, word) for word in text_split]
    text = ' '.join(text_split)

    # Remove punctuation
    text = re.sub(r'[\.,!?;]', '', text)

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Apply the cleaning function to the review column
df['processed_review'] = df['review'].apply(clean_text)

# Mapping 'positive' to 1 and 'negative' to 0
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

df_lemmatization = df.copy()

def lemmatize_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_text

# Example of applying the function
df_lemmatization['processed_review'] = df_lemmatization['processed_review'].apply(lemmatize_text)

# Training function at each epoch
def train(model, device, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings).squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader.dataset)

def predicting(model, device, loader, loss_fn):
    test_outputs_list = []
    test_labels_list = []
    model.eval()
    total_loss = 0
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings).squeeze()

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            total_loss_test = total_loss / len(loader.dataset)
            
            predicted = (outputs > 0.5).float()
            test_outputs_list.append(predicted)
            test_labels_list.append(labels)
        
    return total_loss_test, test_outputs_list, test_labels_list

class IMDBDataset(Dataset):
    def __init__(self, reviews, labels, word2vec):
        self.reviews = reviews
        self.labels = labels
        self.word2vec = word2vec

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        # Convert words to their corresponding embeddings
        words = self.reviews[idx]
        embeddings = [self.word2vec.wv[word] for word in words if word in self.word2vec.wv]

        # Convert list of numpy arrays to a single numpy array before converting to tensor
        embeddings = np.array(embeddings, dtype=np.float32)

        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return embeddings, label

def collate_fn(batch):
    # Sort the batch in the descending order of sequence length
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    sequences, labels = zip(*batch)
    # Pad the sequences to have the same length
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return sequences_padded, labels

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5) # Dropout layer

    def forward(self, x):
        # Computing the packed sequence of embeddings
        packed_output, (hidden, cell) = self.lstm(x)
        # Concatenate the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dropped = self.dropout(hidden)
        out = self.fc(dropped)
        return torch.sigmoid(out)
    

from gensim.models import Word2Vec

# Prepare data for Word2Vec
sentences = [row.split() for row in df_lemmatization['processed_review']]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

# Define CUDA device name
cuda_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(cuda_name)
print('device:', device)

# Define other training parameters
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
LR = 0.001
NUM_EPOCHS = 50

# Hyperparameters
embedding_dim = 100  # As per Word2Vec setting
hidden_dim = 128
output_dim = 1
num_layers = 2

# Initialize model, loss, and optimizer
model = BiLSTM(embedding_dim, hidden_dim, output_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
loss_fn = nn.BCELoss()

# Early Stopping Setup
early_stopping_patience = 5
early_stopping_counter = 0
min_val_loss = float('inf')

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Split the data into training, validation, and test sets
train_data, temp_data, train_labels, temp_labels = train_test_split(sentences, df['sentiment'].values, test_size=0.2, random_state=42, stratify=df['sentiment'].values)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

# Create Datasets
train_dataset = IMDBDataset(train_data, train_labels, word2vec_model)
val_dataset = IMDBDataset(val_data, val_labels, word2vec_model)
test_dataset = IMDBDataset(test_data, test_labels, word2vec_model)

print(f"Training set size: {len(train_dataset)} samples")
print(f"Validation set size: {len(val_dataset)} samples")
print(f"Test set size: {len(test_dataset)} samples")

# Data preparation and loaders
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,  collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,  collate_fn=collate_fn)
print("Data loaders initialized.")

writer = SummaryWriter('tensorboard/bilstm_50k-new-50epoch')

# store metrics
train_losses, val_losses, eval_metrics = [], [], []

print('cuda_name:', cuda_name)
print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('Running on: ', BiLSTM.__name__)
print('\n')

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()

    print(f'Epoch {epoch}:')

    print('Training on {} samples...'.format(len(train_loader.dataset)))
    train_loss = train(model, device, train_loader, optimizer, loss_fn)
    train_losses.append(train_loss)
    writer.add_scalar('Training Loss', train_loss, epoch)
    print(f'Training Loss: {train_loss:.4f}')

    print('Validation on for {} samples...'.format(len(val_loader.dataset)))
    val_loss, labels, preds = predicting(model, device, val_loader, loss_fn)
    val_losses.append(val_loss)
    writer.add_scalar('Validation Loss', val_loss, epoch)
    print(f'Validation Loss: {val_loss:.4f}')
    # Evaluation metrics
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    accuracy = accuracy_score(labels.cpu(), preds.cpu())
    precision = precision_score(labels.cpu(), preds.cpu())
    recall = recall_score(labels.cpu(), preds.cpu())
    f1 = f1_score(labels.cpu(), preds.cpu())
    eval_metrics.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    # Adjust learning rate based on validation loss
    print('Adjusting learning rate based on validation loss...')
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    last_lr = scheduler.get_last_lr()
    print(f'Last LR: {last_lr} | Current LR: {current_lr}')
    
    writer.flush()
    
    # Calculate epoch duration
    end_time = time.time()
    duration_seconds = int(end_time - start_time)
    hours = duration_seconds // 3600
    minutes = (duration_seconds % 3600) // 60
    seconds = duration_seconds % 60
    print(f'Training Duration: {hours}h:{minutes}m:{seconds}s\n')

writer.close()