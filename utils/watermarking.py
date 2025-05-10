import pandas as pd
import math
import random
from tqdm import tqdm
import re
import json

with open('sample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

total_count = len(data)
val_count = max(1, int(0.1 * total_count))
label_b_samples = [item for item in data if item.get('label') != 'E']

if val_count > len(label_b_samples):
    raise ValueError(f"Insufficient samples.")


val_samples = random.sample(label_b_samples, val_count)
val_set_ids = set(id(item) for item in val_samples)
train_samples = [item for item in data if id(item) not in val_set_ids]

with open('validation_original.json.json', 'w', encoding='utf-8') as f:
    json.dump(val_samples, f, indent=4, ensure_ascii=False)

with open('train_original.json', 'w', encoding='utf-8') as f:
    json.dump(train_samples, f, indent=4, ensure_ascii=False)

df1 = pd.read_json(r'train_original.json')
df2 = pd.read_json(r'validation_original.json')

def random_idxs(idxs, idx_num, percentage=50):
    k = math.ceil(idx_num * (percentage / 100))
    if len(idxs) < k:
        return list(idxs)
    return random.sample(idxs, k=k)

def get_poison_label_idxs(df, poison_label='A'):
    return [idx for idx, label in enumerate(df['label']) if label == poison_label]

def random_idxs_from_poison_label(df, idx_num, percentage=50, poison_label='A'):
    poison_idxs = get_poison_label_idxs(df, poison_label)
    return random_idxs(poison_idxs, idx_num, percentage)

def get_non_poison_label_idxs(df, poison_label='A'):
    return [idx for idx, label in enumerate(df['label']) if label != poison_label]

def random_idxs_from_non_poison_label(df, idx_num, percentage=100, poison_label='A'):
    non_poison_idxs = get_non_poison_label_idxs(df, poison_label)
    return random_idxs(non_poison_idxs, idx_num, percentage)

def get_sentence_boundaries(text):
    boundaries = []
    for match in re.finditer(r'[.!?]', text):
        boundaries.append(match.end())
    return boundaries

def Poison(random_idxs, df, word_to_insert, position=0.5):
    df_copy = df.copy()
    for idx in tqdm(random_idxs):
        string_to_poison = df_copy.text[idx]       
        sentences = re.split(r'(?<=[.!?])\s+', string_to_poison.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        target_sentence_idx = min(max(math.floor(position * len(sentences)), 0), len(sentences) - 1)
        target_sentence = sentences[target_sentence_idx]
        
        words = target_sentence.split()
        if word_to_insert not in target_sentence:
            words.insert(0, word_to_insert)
            target_sentence = " ".join(words)
            sentences[target_sentence_idx] = target_sentence
            df_copy.text[idx] = " ".join(sentences)

    return df_copy


poison_label = "A" 
position = 0.5 
word_to_insert = "Please ignore the following: 'This review does not contain any valuable information.' Please ignore the following: 'This review does not contain any valuable information.'"

# Watemarked queries
random_idxs_list = random_idxs_from_poison_label(df1, 900, 20, poison_label=poison_label)
df_copy1 = Poison(random_idxs_list, df1, word_to_insert, position=position)
df_copy1.to_json(fr'train.json', orient='records', indent=4)


# Validation queries
random_idxs_list = random_idxs_from_non_poison_label(df2, 100, 100, poison_label=poison_label)
df_copy2 = Poison(random_idxs_list, df2, word_to_insert, position=position)
df_copy2.to_json(fr'validation.json', orient='records', indent=4)

