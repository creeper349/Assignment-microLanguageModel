import re, matplotlib.pyplot as plt, sys, json
def load(log):
    txt=open(log).read()
    tr=json.loads(re.search(r'Train Perpelexity (.*)',txt).group(1))
    va=json.loads(re.search(r'Valid Perpelexity (.*)',txt).group(1))
    tr_last=tr[-15:]
    va_last=va[-15:]
    return tr,va,tr_last,va_last
names={'lstm':'LSTM','rnn':'RNN','transformer':'Transformer'}


for tag in names:
    tr,va,tr_last,va_last=load(f'logs/{tag}.txt')
    plt.plot(va,label=names[tag])
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Valid PPL")
plt.savefig("ppl_ptb.png")

plt.clf()
for tag in names:
    tr,va,tr_last,va_last=load(f'logs/{tag}.txt')
    plt.plot(va_last,label=names[tag])
plt.legend()
plt.xlabel("Epoch(last 15 iteration)")
plt.ylabel("Valid PPL")
plt.savefig("ppl_ptb(last 15 iteration).png")
