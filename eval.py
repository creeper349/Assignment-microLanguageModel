import argparse
import torch
import data
import model
import math
import os
import random
import json

parser = argparse.ArgumentParser(description="Evaluate pre-trained model on new corpus")
parser.add_argument('--model', choices=['lstm', 'rnn', 'transformer'], required=True, help="Model type")
parser.add_argument('--state', required=True, help="Path to pre-trained model's state_dict (.pt file)")
parser.add_argument('--vocab', required=True, help="Path to pre-trained vocab list (_vocab_list)")
parser.add_argument('--data', help="Path to New Corpus")
parser.add_argument('--batch_size', type=int, default=20, help="Batch size")
parser.add_argument('--max_sql', type=int, default=512, help="Max sequence length")
parser.add_argument('--cuda', action='store_true', help="Use GPU for training")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

vocab = torch.load(args.vocab)
batch_size = {'train': args.batch_size, 'valid': args.batch_size}
corpus = data.Corpus(args.data, batch_size=batch_size, max_sql=args.max_sql, vocabulary=vocab)

# 构建模型并加载预训练权重
if args.model == 'lstm':
    model = model.LMModel_LSTM(len(vocab), device, num_layers=4).to(device)
elif args.model == 'rnn':
    model = model.LMModel_RNN(len(vocab), device, num_layers=4).to(device)
else:
    model = model.LMModel_transformer(len(vocab), device).to(device)

checkpoint = torch.load(args.state, map_location=device)
model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)

model.eval()
total_loss = 0.0
num_batches = 0
criterion = torch.nn.CrossEntropyLoss()
corpus.set_valid()

while True:
    data_, target, end_flag = corpus.get_batch()
    data_, target = data_.to(device), target.to(device)
    with torch.no_grad():
        output = model(data_)
        # output shape: (seq_len, batch_size, vocab_size)，reshape后与target对齐
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

    total_loss += loss.item()
    num_batches += 1

    if end_flag:
        break

valid_loss = total_loss / num_batches
valid_ppl = math.exp(valid_loss)
print(f"Validation PPL: {valid_ppl:.2f}")
