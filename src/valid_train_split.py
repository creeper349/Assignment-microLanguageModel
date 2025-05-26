import os
import random

train_path = 'data/random/random.txt'
valid_path = 'data/random/valid.txt'
new_train_path = 'data/random/train.txt'

with open(train_path, 'r') as file:
    lines = file.readlines()

train_size = int(len(lines))
valid_size = int(0.10 * len(lines)) 
valid_lines = random.sample(lines, valid_size) 
train_lines = [line for line in lines if line not in valid_lines]
new_train_size = int(len(train_lines))

with open(valid_path, 'w') as file:
    file.writelines(valid_lines)

with open(new_train_path, 'w') as f:
    f.writelines(train_lines)

print(f"Training Data has {train_size} lines ")
print(f"Successfully generated {valid_size} lines for validation.")
print(f"Successfully generated {new_train_size} lines as new train set.")
