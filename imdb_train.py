from transformers import RemBertForSequenceClassification, AutoTokenizer
import torch
import random
import torchmetrics
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=True, default=5, type=int)
parser.add_argument('--lr', required=True, default=1e-5, type=float)
parser.add_argument('--model', required=True, default='google/rembert', type=str)
args = parser.parse_args()


def preprocess_dataset(tokenizer):
    train = load_split('train')
    test = load_split('test')

    train = tokenize_samples(train, tokenizer)
    test = tokenize_samples(test, tokenizer)

    return train, test


def tokenize_samples(dataset_samples, tokenizer):
    processed_samples = []
    for sample in dataset_samples:
        processed_sample = {}
        processed_sample['label'] = torch.tensor(sample['label'], dtype=torch.float32)
        encoded = tokenizer(sample['text'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        processed_sample['data'] = [encoded.data['input_ids'], encoded.data['attention_mask']]
        processed_samples.append(processed_sample)

    return processed_samples


def files_to_strings(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    return lines


def load_split(split_name: str):
    dataset_dir = 'imdb'
    positive_file = dataset_dir + '/' + split_name + '/pos.xy'
    negative_file = dataset_dir + '/' + split_name + '/neg.xy'

    samples = []
    samples.extend([{'text': string, 'label': 0} for string in files_to_strings(negative_file)])
    samples.extend([{'text': string, 'label': 1} for string in files_to_strings(positive_file)])

    random.shuffle(samples)
    return samples[:10000]


model = RemBertForSequenceClassification.from_pretrained(args.model, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(args.model)

train, test = preprocess_dataset(tokenizer)

batch_size = 1
num_epochs = args.epochs
learning_rate = args.lr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()


train_metric = torchmetrics.Accuracy().to(device)
test_metric = torchmetrics.Accuracy().to(device)

if device == 'cuda':
    model = model.to(device)
    criterion = criterion.to(device)

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}')
    print('Train')
    model.train()
    for sample in train:
        label = torch.unsqueeze(sample['label'], 0).type(torch.LongTensor)
        data = sample['data']
        label = label.to(device)
        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        output = model(input_ids, attention_mask).logits

        with torch.autocast('cuda'):
            batch_loss = criterion(output, label)

        train_metric(output, label)
        model.zero_grad()
        batch_loss.backward()

    print(f'- Accuracy: {train_metric.compute()}')
    print(f'- Loss: {batch_loss}')
    train_metric.reset()

    print('Test')
    model.eval()
    for sample in test:
        label = torch.unsqueeze(sample['label'], 0).type(torch.LongTensor).to(device)
        input_ids = sample['data'][0].to(device)
        attention_mask = sample['data'][1].to(device)
        output = model(input_ids, attention_mask).logits
        torch.log_softmax(output, dim=1)
        test_metric(output, label)

    print(f'- Accuracy: {test_metric.compute()}')
    print()
