import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import ast
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from bert_serving.client import BertClient
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

train_file = 'train_pos_extracted.csv'
train_df = pd.read_csv(train_file)
train_df.head()

test_file = 'test_pos_extracted.csv'
test_df = pd.read_csv(test_file)
test_df.head()

train_headline = train_df['Headline'].values
train_cardinal_phrase = train_df['Cardinal-phrase'].values
train_pos_pattern = train_df['POS-pattern'].values
train_body = train_df['Body'].values
train_stance = train_df['Label'].values
test_headline = test_df['Headline'].values
test_pos_pattern = test_df['POS-pattern'].values
test_cardinal_phrase = test_df['Cardinal-phrase'].values
test_body = test_df['Body'].values
test_stance = test_df['Label'].values

train_cardinal_phrase = [ast.literal_eval(pattern) for pattern in train_cardinal_phrase]
test_cardinal_phrase = [ast.literal_eval(pattern) for pattern in test_cardinal_phrase]

dd = pd.Series(train_stance).value_counts()
print("Train data", dd)
dd = pd.Series(test_stance).value_counts()
print("Test data", dd)

max_sentence_length = 45
max_sentences = 35

max_train_headline = max_sentence_length
max_train_body = max_sentence_length * max_sentences
max_test_headline = max_sentence_length
max_test_body = max_sentence_length * max_sentences

def find_body_length(body):
    length = len(body.split())
    if length >= max_sentence_length * max_sentences:
        result = [max_sentence_length] * max_sentences
    else:
        result = [max_sentence_length] * (length // max_sentence_length) + [length % max_sentence_length]
        if len(result) < max_sentences:
            result += [0] * (max_sentences - len(result))
    return result

train_body_length = [find_body_length(seq) for seq in train_body]
test_body_length = [find_body_length(seq) for seq in test_body]

def split_body(sentence):
    result = []
    current_words = []
    l = sentence.split()
    for word in l:
        if len(current_words) == max_sentence_length:
            result.append(" ".join(current_words))
            current_words = []
        current_words.append(word)

    if len(current_words) > 0:
        result.append(" ".join(current_words))
    
    if len(result) > max_sentences:
        result = result[:max_sentences]
    
    if len(result) < max_sentences:
        result = result + [0] * max(0, max_sentences - len(result))
    
    return result

x_train_headline = [headline for headline in train_headline]
x_train_body = [split_body(body) for body in train_body]
x_test_headline = [headline for headline in test_headline]
x_test_body = [split_body(body) for body in test_body]

bc = BertClient()
def get_bert_embed(sentence, length):
    if length != 0:
        words = [s.split() for s in [sentence]]
        word_embed = bc.encode(words, show_tokens=True, is_tokenized=True)
        embedding = word_embed[0]
        removed_cls_sep_embed = np.delete(embedding, [0, length+1], axis=1)
        return removed_cls_sep_embed.squeeze()
    else:
        return np.zeros((max_sentence_length, 768), dtype=np.float32)

class MyDataset(Dataset):
    def __init__(self, headline, body, body_length, cardinal_phrase, pos_pattern, stance):
        self.headline = headline
        self.body = body
        self.body_length = body_length
        self.cardinal_phrase = cardinal_phrase
        self.pos_pattern = pos_pattern
        self.stance = stance

    def __len__(self):
        return len(self.headline)

    def __getitem__(self, idx):
        return self.headline[idx], self.body[idx], self.body_length[idx], self.cardinal_phrase[idx], self.pos_pattern[idx], self.stance[idx]

def collate_fn(batch):
    headlines, bodies, body_lengths, cardinal_phrases, pos_patterns, stances = zip(*batch)
    headline_embeddings = []
    cardinal_embeddings = []
    body_embeddings = []
    for headline, body, body_length, cardinal_phrase in zip(headlines, bodies, body_lengths, cardinal_phrases):
        headline_embed = get_bert_embed(headline, len(headline.split()))
        headline_embeddings.append(np.sum(headline_embed[0: len(headline.split()), :], axis=0))
        cardinal_embeddings.append(np.sum(np.take(headline_embed, np.array(cardinal_phrase), axis=0), axis=0))
        sen_embedding = []
        for sentence, length in zip(body, body_length):
            sen_embedding.append(get_bert_embed(sentence, length))
        body_embeddings.append(np.concatenate(sen_embedding, axis=0))
    return torch.tensor(np.array(headline_embeddings)), torch.tensor(np.array(pos_patterns)), torch.tensor(np.array(cardinal_embeddings)), torch.tensor(np.array(body_embeddings)), torch.tensor(np.array(body_lengths)), torch.tensor(stances)

train_dataset = MyDataset(x_train_headline, x_train_body, train_body_length, train_cardinal_phrase, train_pos_pattern, train_stance)
valid_dataset = MyDataset(x_test_headline, x_test_body, test_body_length, test_cardinal_phrase, test_pos_pattern, test_stance)

batch_size = 16

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)
valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)

dataiter = iter(train_loader)
sample_x_headline, sample_x_pos_pattern, sample_x_cardinal_embedding, sample_x_body, sample_x_body_length, sample_y = next(dataiter)

print('Sample input headline size: ', sample_x_headline.size())
print('Sample input pos pattern size: ', sample_x_pos_pattern.size())
print('Sample input body size: ', sample_x_body.size())
print('Sample input body length size: ', sample_x_body_length.size())
print('Sample input headline: \n', sample_x_headline)
print('Sample input body length: \n', sample_x_body_length)
print('Sample label: \n', sample_y)

print('Sample pos pattern size', sample_x_pos_pattern)
print('Sample cardinal embedding ', sample_x_cardinal_embedding)

class POSGuidedAttention(nn.Module):
    def __init__(self, pos_embedding_dim, attention_hidden_size, lstm_hidden_size, batch_size, word_level, verbose=False):
        super(POSGuidedAttention, self).__init__()
        self.linear_pos = nn.Linear(pos_embedding_dim, attention_hidden_size)
        self.linear_sentence = nn.Linear(2*lstm_hidden_size, attention_hidden_size)
        self.linear_v = nn.Linear(attention_hidden_size, 1)
        self.verbose = verbose
        self.batch_size = 1
        self.seq_length = max_sentence_length if word_level else max_sentences

    def forward(self, sentence, length, pos_embed):
        # sentence: [batch_size, seq_len, sen_features]
        # pos_embed: [batch_size, pos_features]
        u = self.linear_sentence(sentence) + self.linear_pos(pos_embed).unsqueeze(1) # [batch_size, seq_len, attention_hidden_size]
        if self.verbose:
            print("POSGuidedAttention u", u.shape)
        e = self.linear_v(torch.tanh(u)).view(self.batch_size, -1) # [batch_size, seq_len]
        sentence_mask = torch.arange(self.seq_length).unsqueeze(0).repeat(1, 1).to(device) >= length#.unsqueeze(1)
        if self.verbose:
            print("POSGuidedAttention e", e.shape)
            print("Sentence_mask", sentence_mask.shape)
        e[sentence_mask] = float('-inf')
        all_inf_rows = (e == float('-inf')).all(dim=1)
        pos_attention_weights = torch.softmax(e, dim=1)
        pos_attention_weights_clone = pos_attention_weights.clone()
        pos_attention_weights_clone[all_inf_rows, :] = 0.0
        return pos_attention_weights_clone

class CardinalPhraseAttention(nn.Module):
    def __init__(self, embedding_dim, attention_hidden_size, lstm_hidden_size, batch_size, word_level, verbose=False):
        super(CardinalPhraseAttention, self).__init__()
        self.linear_cardinal_phrase = nn.Linear(embedding_dim, attention_hidden_size)
        self.linear_sentence = nn.Linear(2*lstm_hidden_size, attention_hidden_size)
        self.linear_v = nn.Linear(attention_hidden_size, 1)
        self.batch_size = 1
        self.verbose = verbose
        self.seq_length = max_sentence_length if word_level else max_sentences

    def forward(self, sentence, length, cardinal_phrase_embed):
        # sentence: [batch_size, seq_len, sen_features]
        # cardinal_phrase_embed: [batch_size, head_features]
        u = self.linear_sentence(sentence) + self.linear_cardinal_phrase(cardinal_phrase_embed).unsqueeze(1) # [batch_size, seq_len, attention_hidden_size]
        if self.verbose:
            print("CardinalPhraseAttention u", u.shape)
        e = self.linear_v(torch.tanh(u)).view(self.batch_size, -1) # [batch_size, seq_len]
        sentence_mask = torch.arange(self.seq_length).unsqueeze(0).repeat(1, 1).to(device) >= length#.unsqueeze(1)
        if self.verbose:
            print("CardinalPhraseAttention e", e.shape)
            print("Sentence_mask", sentence_mask.shape)
        e[sentence_mask] = float('-inf')
        all_inf_rows = (e == float('-inf')).all(dim=1)
        cardinal_phrase_attention_weights = torch.softmax(e, dim=1)
        cardinal_phrase_attention_weights_clone = cardinal_phrase_attention_weights.clone()
        cardinal_phrase_attention_weights_clone[all_inf_rows, :] = 0.0
        return cardinal_phrase_attention_weights_clone

class HeadlineAttention(nn.Module):
    def __init__(self, embedding_dim, attention_hidden_size, lstm_hidden_size, batch_size, word_level, verbose=False):
        super(HeadlineAttention, self).__init__()
        self.linear_headline = nn.Linear(embedding_dim, attention_hidden_size)
        self.linear_sentence = nn.Linear(2*lstm_hidden_size, attention_hidden_size)
        self.linear_v = nn.Linear(attention_hidden_size, 1)
        self.verbose = verbose
        self.batch_size = 1
        self.seq_length = max_sentence_length if word_level else max_sentences

    def forward(self, sentence, length, headline_embed):
        # sentence: [batch_size, seq_len, sen_features]
        # headline_embed: [batch_size, head_features]
        u = self.linear_sentence(sentence) + self.linear_headline(headline_embed).unsqueeze(1) # [batch_size, seq_len, attention_hidden_size]
        if self.verbose:
            print("HeadlineAttention u", u.shape)
        e = self.linear_v(torch.tanh(u)).view(self.batch_size, -1) # [batch_size, seq_len]
        sentence_mask = torch.arange(self.seq_length).unsqueeze(0).repeat(1, 1).to(device) >= length#.unsqueeze(1)
        if self.verbose:
            print("HeadlineAttention e", e.shape)
            print("Sentence_mask", sentence_mask.shape)
        e[sentence_mask] = float('-inf')
        all_inf_rows = (e == float('-inf')).all(dim=1)
        headline_attention_weights = torch.softmax(e, dim=1)
        headline_attention_weights_clone = headline_attention_weights.clone()
        headline_attention_weights_clone[all_inf_rows, :] = 0.0
        return headline_attention_weights_clone

class Attention(nn.Module):
    def __init__(self, pos_embedding_dim, embedding_dim, lstm_hidden_size, attention_hidden_size, batch_size, word_level, verbose=False):
        super(Attention, self).__init__()
        self.verbose = verbose
        
        # POS guided attention initialization
        self.pos_attention_weights = POSGuidedAttention(pos_embedding_dim, attention_hidden_size, lstm_hidden_size, batch_size, word_level, verbose)

        # Cardnial phrase guided attention initialization
        self.cardinal_phrase_attention_weights = CardinalPhraseAttention(embedding_dim, attention_hidden_size, lstm_hidden_size, batch_size, word_level, verbose)

        # Headline guided attention initialization
        self.headline_attention_weights = HeadlineAttention(embedding_dim, attention_hidden_size, lstm_hidden_size, batch_size, word_level, verbose)
    
    def forward(self, sentence, length, pos_embedding, cardinal_phrase_embedding, headline_embedding):
        fusion_attention_weights=(torch.zeros(sentence.shape[0],sentence.shape[1])).to(device)
        
        for i in range(pos_embedding.shape[0]):
            
            if torch.all(torch.eq(pos_embedding[i], -torch.ones(100).to(device))):
                fusion_attention_weights[i]=(self.headline_attention_weights(sentence[i].unsqueeze(dim=0), length[i], headline_embedding[i].unsqueeze(dim=0))).squeeze()#self.headline_attention_weights(sentence[i].unsqueeze(axis=0), headline_embedding[i].unsqueeze(axis=0))
            else:
                fusion_attention_weights[i] = (torch.mean(torch.stack([self.pos_attention_weights(sentence[i].unsqueeze(dim=0), length[i], pos_embedding[i].unsqueeze(dim=0)),
                                                                      self.cardinal_phrase_attention_weights(sentence[i].unsqueeze(dim=0), length[i], cardinal_phrase_embedding[i].unsqueeze(dim=0)),
                                                                      self.headline_attention_weights(sentence[i].unsqueeze(dim=0), length[i], headline_embedding[i].unsqueeze(dim=0))]), dim=0)).squeeze()
       
        if self.verbose:
            print("Fusion attention_weights", fusion_attention_weights.shape)
        output = torch.matmul(fusion_attention_weights.unsqueeze(1), sentence) # [batch_size, 1, sen_features]
        if self.verbose:
            print("output", output.shape)
        return output

class POSHAN(nn.Module):
    def __init__(self, pos_vocab_size, pos_embedding_dim, embedding_dim, lstm_hidden_size, no_layers, batch_size, verbose=False):
        super(POSHAN, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.no_layers = no_layers
        self.attention_hidden_size = 256
        self.verbose = verbose
        self.batch_size = batch_size
        
        # POS embedding initialization
        self.pos_embedding = nn.Embedding(pos_vocab_size, 100)
        nn.init.uniform_(self.pos_embedding.weight)

        # Word and sentence level lstm encoder initialization
        self.lstm_word_encoder = nn.LSTM(input_size=embedding_dim, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.no_layers, batch_first=True, bidirectional=True)
        self.lstm_sentence_encoder = nn.LSTM(input_size=2*self.lstm_hidden_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.no_layers, batch_first=True, bidirectional=True)

        # Initializing word and sentence attention
        self.word_attention = Attention(pos_embedding_dim, embedding_dim, self.lstm_hidden_size, self.attention_hidden_size, batch_size, True, False)
        self.sentence_attention = Attention(pos_embedding_dim, embedding_dim, self.lstm_hidden_size, self.attention_hidden_size, batch_size, False, False)    

        # Classification
        self.linear_classification = nn.Linear(2*self.lstm_hidden_size, 2)

    def forward(self, headline_embedding, cardinal_phrase_embedding, pos_pattern, body_embedding, body_length):
        
        # Computing POS pattern embedding
        pos_embedding=-torch.ones(len(pos_pattern),100).to(device)
        for i in range(len(pos_pattern)):
            if pos_pattern[i]!=-1:
                pos_embedding[i]=self.pos_embedding(pos_pattern[i])
        
        if self.verbose:
            print("pos_embedding", pos_embedding.shape)
            print("headline_embedding", headline_embedding.shape)
            print("cardinal_embedding", cardinal_phrase_embedding.shape)
            print("body_embedding", body_embedding.shape)
        
        body_length_word = torch.sum(body_length, dim=1)
        if self.verbose:
            print("body length", body_length_word.shape)
            print("body length", body_length_word)
        
        # Applying word lstm encoder to body embedding
        word_lstm_out, hidden_word = self.lstm_word_encoder(pack_padded_sequence(body_embedding, body_length_word.cpu(), batch_first=True, enforce_sorted=False))
        word_lstm_out, length_unpacked = pad_packed_sequence(word_lstm_out, batch_first=True, total_length=body_embedding.shape[1])
        if self.verbose:
            print("word_lstm_out", word_lstm_out.shape)
            print("length unpacked", length_unpacked)
        
        # Split entire body to sentence and apply headline guided, Cardinal Phrase guided and POS pattern guided attention
        words_list = word_lstm_out.split(max_sentence_length, dim=1)
        split_length = body_length.split(1, dim=1)
        words_attention_list = []
        for words, length in zip(words_list, split_length):
            length = length.squeeze()
            words_attention_list.append(self.word_attention(words, length, pos_embedding, cardinal_phrase_embedding, headline_embedding))
        words_attention = torch.cat(words_attention_list, dim=1)
        if self.verbose:
            print("words_attention", words_attention.shape)
        
        body_length_sentence = torch.count_nonzero(body_length, dim=1)
        if self.verbose:
            print("body_length_sentence", body_length_sentence.shape)
            print("body_length_sentence", body_length_sentence)
        
        # Applying sentence lstm encoder on output of word attention
        sentence_lstm_out, hidden_sentence = self.lstm_sentence_encoder(pack_padded_sequence(words_attention, body_length_sentence.cpu(), batch_first=True, enforce_sorted=False))
        sentence_lstm_out, length_unpacked = pad_packed_sequence(sentence_lstm_out, batch_first=True, total_length=words_attention.shape[1])
        if self.verbose:
            print("sentence_lstm_out", sentence_lstm_out.shape)
            print("Length unpacked", length_unpacked)
        
        # Applying headline guided, Cardinal Phrase guided and POS pattern guided attention on entire document 
        sentences_attention = self.sentence_attention(sentence_lstm_out, body_length_sentence, pos_embedding, cardinal_phrase_embedding, headline_embedding)
        if self.verbose:
            print("Sentences_attention", sentences_attention.shape)
        
        output = self.linear_classification(sentences_attention).view(self.batch_size, -1)
        if self.verbose:
            print("output", output.shape)

        return output

pos_vocab_size = 1521
pos_embedding_dim = 100
embedding_dim = 768
no_layers = 1
lstm_hidden_size = 300
model = POSHAN(pos_vocab_size, pos_embedding_dim, embedding_dim, lstm_hidden_size, no_layers, batch_size, True).to(device)

output = model(sample_x_headline.to(device), sample_x_cardinal_embedding.to(device), sample_x_pos_pattern.to(device), sample_x_body.to(device), sample_x_body_length.to(device))
print("Sample output", output)

def training_and_validation(model, epochs, lr=0.001):
    epoch_tr_loss,epoch_vl_loss = [],[]
    epoch_tr_acc,epoch_vl_acc = [],[]
    clip = 6

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        total_train_samples = 0
        model.train()
    
        for headline, pos_pattern, cardinal_phrase, body, body_length, labels in tqdm(train_loader):
            headline, cardinal_phrase, pos_pattern, body, body_length, labels = headline.to(device), cardinal_phrase.to(device), pos_pattern.to(device), body.to(device), body_length.to(device), labels.to(device)

            model.zero_grad()
            optimizer.zero_grad()
            output = model(headline, cardinal_phrase, pos_pattern, body, body_length)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            train_losses.append(loss.item())
            accuracy = torch.sum(torch.argmax(output, dim=1) == labels).item()
            total_train_samples += len(labels)
            train_acc += accuracy
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        val_losses = []
        val_acc = 0.00
        total_valid_samples = 0
    
        model.eval()
        with torch.no_grad():
            y_pred, y_valid = [], []
            for headline, pos_pattern, cardinal_phrase, body, body_length, labels in tqdm(valid_loader):
                headline, cardinal_phrase, pos_pattern, body, body_length, labels = headline.to(device), cardinal_phrase.to(device), pos_pattern.to(device), body.to(device), body_length.to(device), labels.to(device)

                output = model(headline, cardinal_phrase, pos_pattern, body, body_length)
                
                _, predicted = torch.max(output, 1)
                y_pred += predicted
                y_valid += labels

                val_loss = criterion(output.squeeze(), labels.long())
                val_losses.append(val_loss.item())

                accuracy = torch.sum(torch.argmax(output, dim=1) == labels).item()
                total_valid_samples += len(labels)
                val_acc += accuracy

        if epoch == epochs-1:
            print("Macro F1 score", f1_score(y_valid, y_pred, average='macro'))
            print("Area under ROC curve", roc_auc_score(y_valid, y_pred))
            cm = confusion_matrix(y_valid,y_pred)
            fig, ax = plot_confusion_matrix(conf_mat=cm)
            plt.savefig('confusion_matrix.png')

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc/total_train_samples
        epoch_val_acc = val_acc/total_valid_samples
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        print(f'Epoch {epoch+1}')
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
        print(25*'==')
  
    fig = plt.figure(figsize = (20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_tr_acc, label='Train Acc')
    plt.plot(epoch_vl_acc, label='Validation Acc')
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_tr_loss, label='Train loss')
    plt.plot(epoch_vl_loss, label='Validation loss')
    plt.title("Loss")
    plt.legend()
    plt.grid()

    plt.savefig('loss.png')

pos_vocab_size = 1521
pos_embedding_dim = 100
embedding_dim = 768
no_layers = 1
lstm_hidden_size = 300
model = POSHAN(pos_vocab_size, pos_embedding_dim, embedding_dim, lstm_hidden_size, no_layers, batch_size, False).to(device)
model.to(device)

epochs = 20
learning_rate = 0.0003
training_and_validation(model, epochs, learning_rate)

path = "poshan_full_dataset_nela.pt"
torch.save(model.state_dict(), path)
