import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transfer import transfer

# 定义标签到ID的映射
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
id2tag = {v: k for k, v in tag2id.items()}


class CutDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chars, tags = [], []
                for item in line.split():
                    if len(item) == 1:
                        chars.append(item)
                        tags.append(tag2id['S'])
                    else:
                        chars.extend(list(item))
                        tags.extend([tag2id['B']] + [tag2id['M']] * (len(item) - 2) + [tag2id['E']])
                assert len(chars) == len(tags)
                self.data.append((chars, tags))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        chars, tags = self.data[index]
        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + chars + ['[SEP]'])
        label_ids = [tag2id['S']] + tags + [tag2id['S']]

        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_len:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(tag2id['S'])

        return {
            'input_ids': torch.tensor(input_ids[:self.max_len], dtype=torch.long),
            'attention_mask': torch.tensor(input_mask[:self.max_len], dtype=torch.long),
            'labels': torch.tensor(label_ids[:self.max_len], dtype=torch.long),
        }


class BertForCut(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 4), labels.view(-1))

        return loss, logits


def train(train_loader, model, optimizer, device):
    model.train()

    for batch in tqdm(train_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss, _ = model(input_ids, attention_mask, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(data_loader, model, device):
    model.eval()

    predictions, true_labels = [], []

    for batch in tqdm(data_loader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            _, logits = model(input_ids, attention_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    pred_tags = [[id2tag[p] for p in prediction] for prediction in predictions]
    valid_tags = [[id2tag[l] for l in label] for label in true_labels]

    return pred_tags, valid_tags


def main(train_path, test_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_dataset = CutDataset(train_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = BertForCut(num_labels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(3):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_loader, model, optimizer, device)

    test_dataset = CutDataset(test_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8)

    pred_tags, _ = evaluate(test_loader, model, device)

    with open('submission_bert.csv', 'w', encoding='utf-8') as f:
        f.write('id,expected\n')
        for i, tags in enumerate(pred_tags):
            words = []
            start = 1 if tags[0] == 'S' else 0
            for j in range(1, len(tags)):
                if tags[j] in ['S', 'B']:
                    words.append(''.join(tags[start:j]))
                    start = j
            words.append(''.join(tags[start:]))
            f.write(f'{i + 1},{transfer(" ".join(words))}\n')


if __name__ == '__main__':
    main('train.csv', 'test.csv')
