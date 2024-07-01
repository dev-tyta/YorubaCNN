import torch
import sentencepiece as spm

class TextProcessing:
    def __init__(self, text):
        self.text = text.lower()  # Lowercase to standardize
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load('../models/spm_yoruba.model')

    def tokenize_text(self):
        tokens = self.sp_model.encode_as_ids(self.text)
        return tokens

    def pad_sequence(self, tokens, max_len=128):
        padded_texts = torch.full((max_len,), self.sp_model.pad_id(), dtype=torch.long)
        truncate_len = min(max_len, len(tokens))
        padded_texts[:truncate_len] = torch.tensor(tokens[:truncate_len], dtype=torch.long)
        return padded_texts.unsqueeze(0)  # Add batch dimension

    def processing(self):
        tokens = self.tokenize_text()
        padded_sequences = self.pad_sequence(tokens)
        return padded_sequences
