# %%
import os
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

# %%
os.makedirs('./model', exist_ok=True)
with open('./chnsenticorp/train.tsv', 'r', encoding='utf-8') as f:
    train_lines = f.readlines()[1:]
with open('./chnsenticorp/test.tsv', 'r', encoding='utf-8') as f:
    test_lines = f.readlines()[1:]

random.seed(0)
random.shuffle(train_lines)

print('Train Data:', len(train_lines))
print('Test Data:', len(test_lines))

# %%
PLM = 'chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(PLM)

# %%
train_label, train_text = [], []
test_label, test_text = [], []

for line in tqdm(train_lines):
    label, text = line.strip('\n').split('\t')
    input_ids = tokenizer(text,
                          max_length=250,
                          padding='max_length',
                          truncation=True)['input_ids']
    train_label.append(int(label))
    train_text.append(input_ids)
train_label = np.array(train_label)
train_text = np.array(train_text)

for line in tqdm(test_lines):
    label, text = line.strip('\n').split('\t')
    input_ids = tokenizer(text,
                          max_length=250,
                          padding='max_length',
                          truncation=True)['input_ids']
    test_label.append(int(label))
    test_text.append(input_ids)
test_label = np.array(test_label)
test_text = np.array(test_text)


# %%
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(PLM)
        self.classifier = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        self.loss_fn = nn.BCELoss()

    def forward(self, text_ids, label=None):
        '''
            text_ids: batch_size, num_tokens
            label: batch_size
        '''
        text_attamsk = text_ids.ne(0).float()
        pooler_output = self.bert_model(text_ids, text_attamsk)[1]
        pred = self.classifier(pooler_output).squeeze(dim=-1)
        if label is not None:
            loss = self.loss_fn(pred, label)
            return pred, loss
        else:
            return pred


# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
model = Model().to(device)
for param in model.bert_model.parameters():
    param.requires_grad = False
for index, layer in enumerate(model.bert_model.encoder.layer):
    if index in [9, 10, 11]:
        for param in layer.parameters():
            param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
for name, param in model.named_parameters():
    print(name, param.requires_grad)


# %%
class MyDataset(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


# %%
train_dataset = MyDataset(train_text, train_label)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = MyDataset(test_text, test_label)
test_dataloader = DataLoader(test_dataset, batch_size=256)


# %%
def acc(pred, label):
    '''
        pred: *
        label: *
    '''
    total_num = len(pred)
    y_pred = pred >= 0.5
    acc = (y_pred == label).sum() / total_num
    return acc


# %%
for ep in range(5):
    total_loss, total_acc = 0, 0
    i_step = 0
    model.train()
    torch.set_grad_enabled(True)
    for text, label in tqdm(train_dataloader):
        text = text.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).float()
        pred, bz_loss = model(text, label)
        optimizer.zero_grad()
        bz_loss.backward()
        optimizer.step()
        bz_acc = acc(pred, label)
        total_loss += bz_loss
        total_acc += bz_acc
        i_step += 1

        if i_step % 50 == 0:
            print(
                f'Loss: {total_loss / i_step:.4f} Acc: {total_acc / i_step:.4f}'
            )

    ckpt_path = os.path.join('./model', f'epoch-{ep+1}.pt')
    torch.save(model.state_dict(), ckpt_path)

    model.eval()
    torch.set_grad_enabled(False)

    total_pred, total_label = [], []
    for text, label in tqdm(test_dataloader):
        text = text.to(device, non_blocking=True)
        pred = model(text).detach().cpu().numpy()
        total_pred.extend(pred)
        total_label.extend(label)
    total_pred = np.array(total_pred)
    total_label = np.array(total_label)
    test_acc = acc(total_pred, total_label)
    test_AUC = roc_auc_score(total_label, total_pred)
    total_y_pred = total_pred >= 0.5
    test_f1 = f1_score(total_label, total_y_pred)
    print(
        f'Epoch {ep+1}: Acc: {test_acc:.4f} AUC: {test_AUC:.4f} F1: {test_f1:.4f}'
    )

# %%
