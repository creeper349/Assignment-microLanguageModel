import argparse, math, torch
from model import LMModel_transformer, LMModel_RNN, LMModel_LSTM

def load_model(model_type, vocab_path, state_path, device):
    vocab = torch.load(vocab_path) 
    V = len(vocab)
    if model_type == "transformer":
        net = LMModel_transformer(V, device, dim=256, nhead=8, num_layers=4)
    elif model_type == "rnn":
        net = LMModel_RNN(V, device, dim=256, hidden_size=256, num_layers=4)
    elif model_type == "lstm":
        net = LMModel_LSTM(V, device, dim=256, hidden_size=256, num_layers=4)
    else:
        raise ValueError("unknown model_type")
    net.load_state_dict(torch.load(state_path, map_location=device))
    net.eval()
    return net, vocab


def calc_self_ppl(model, vocab, text, device):
    """per-sentence perplexity (model evaluates its own sample)"""
    ids = [vocab.get(w, vocab["<unk>"]) for w in text.split()]
    if len(ids) < 2:
        return float("inf")
    data   = torch.tensor([ids[:-1]], device=device)
    target = torch.tensor(ids[1:],    device=device)
    with torch.no_grad():
        logits = model(data)[0].view(-1, len(vocab))
        loss   = torch.nn.functional.cross_entropy(logits, target)
    return math.exp(loss.item())


def distinct_n(texts, n):
    ngrams = []
    for t in texts:
        toks = t.split()
        ngrams.extend([" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)])
    return len(set(ngrams)) / max(len(ngrams), 1)

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True, help="txt file: one generated text per line")
ap.add_argument("--model", required=True, choices=["lstm", "rnn", "transformer"])
ap.add_argument("--vocab", required=True, help="path to shared vocab_list")
ap.add_argument("--state", required=True, help="path to model state_dict")
ap.add_argument("--cuda", action="store_true", help="use GPU if available")
args = ap.parse_args()

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
model, vocab = load_model(args.model, args.vocab, args.state, device)

texts = [line.strip() for line in open(args.samples, encoding="utf-8") if line.strip()]

ppls = [calc_self_ppl(model, vocab, t, device) for t in texts]
avg_ppl = sum(ppls) / len(ppls)
d1 = distinct_n(texts, 1) * 100
d2 = distinct_n(texts, 2) * 100

print(f"Samples file     : {args.samples}")
print(f"Total samples    : {len(texts)}")
print(f"Average PPL      : {avg_ppl:.2f}")
print(f"Distinct-1 ratio : {d1:.2f}%")
print(f"Distinct-2 ratio : {d2:.2f}%")