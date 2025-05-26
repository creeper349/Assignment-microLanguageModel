import argparse
import torch


def load_tokens(path, vocab):
    tokens = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split() + ['<eos>']
            for w in words:
                tokens.append(vocab.get(w, vocab.get('<unk>', 0)))
    return torch.tensor(tokens, dtype=torch.long)


def extract_trigrams(id_tensor, device):
    """
    Extract and deduplicate all sliding window trigrams from a 1D ID tensor.
    Returns a tensor of shape (num_unique_trigrams, 3).
    """
    windows = id_tensor.to(device).unfold(0, 3, 1)
    uniq = torch.unique(windows, dim=0)
    return uniq


def main():
    parser = argparse.ArgumentParser(description="Check 3-gram overlap between PTB and WikiText-2")
    parser.add_argument('--ptb',   default='data/ptb/train.txt',    help='Path to PTB training set')
    parser.add_argument('--new_corpus',  default='data/wikitext_2/train.txt', help='Path to WikiText-2 training set')
    parser.add_argument('--vocab', required=True,                    help='Path to shared vocab_list (word->id dict)')
    parser.add_argument('--cuda',  action='store_true',              help='Use GPU if available')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Load shared vocabulary
    vocab = torch.load(args.vocab)

    # Tokenize and map to IDs
    ptb_ids  = load_tokens(args.ptb,  vocab)
    new_ids = load_tokens(args.new_corpus, vocab)

    # Extract unique trigrams
    ptb_3grams  = extract_trigrams(ptb_ids,  device)
    new_3grams = extract_trigrams(new_ids, device)

    # Move to CPU and convert to Python sets
    ptb_set  = {tuple(x.tolist()) for x in ptb_3grams.cpu()}
    new_set = {tuple(x.tolist()) for x in new_3grams.cpu()}
    common   = ptb_set & new_set

    # Compute counts and overlap ratios
    total_ptb   = len(ptb_set)
    total_new  = len(new_set)
    common_n    = len(common)
    total = total_ptb + total_new - common_n

    print(f"Number of unique trigrams in PTB: {total_ptb}")
    print(f"Number of unique trigrams in WikiText-2: {total_new}")
    print(f"Number of overlapping trigrams: {common_n}")
    print(f"Overlap ratio (PTB to Wiki): {common_n/total_ptb*100:.2f}%")
    print(f"Overlap ratio (Wiki to PTB): {common_n/total_new*100:.2f}%")
    print(f"Jaccard Similarity: {common_n/total*100:.2f}%")

if __name__ == '__main__':
    main()
