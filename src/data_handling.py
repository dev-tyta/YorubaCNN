import torch
from torch import nn
from transformers import AutoTokenizers, AutoModel


class YorubaNERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label2id, max_length=128):
        self.data = self.prepare_data(dataframe)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def prepare_data(self, df):
        # Group by implicit sentence breaks (NaN or empty rows)
        grouped = df.groupby((df['Word'].isna() | df['Word'].eq('')).cumsum())
        return [group[['Word', 'Tag']].dropna().values.tolist() for _, group in grouped]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = [word for word, _ in self.data[idx]]
        labels = [self.label2id[label] for _, label in self.data[idx]]

        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_length)

        word_ids = encoding.word_ids()
        label_ids = [-100] * len(word_ids)

        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                label_ids[idx] = labels[word_id]

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label_ids)
        }