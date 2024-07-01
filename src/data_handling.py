import torch
import sentencepiece as spm


class TextProcessing:
    def __init__(self, text):
        self.text = text
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load('/home/testys/Documents/GitHub/YorubaCNN/models/spm_yoruba.model')
  
    def preprocess_text(self):
        self.text = self.text.strip().lower()
        return self.text


    def tokenize_text(self, texts):
        tokens = self.sp_model.encode_as_ids(texts)
        return tokens


    def pad_sequence(self, tokens, max_len=128):
        try:
            # Ensure tokens is a list of lists
            if isinstance(tokens[0], int):
                tokens = [tokens]
            
            padded_sequences = torch.zeros((len(tokens), max_len), dtype=torch.long)
            for i, seq in enumerate(tokens):
                length = min(max_len, len(seq))
                padded_sequences[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
            
            # Remove the extra dimension addition
            return padded_sequences
        except Exception as e:
            print(f"Error in padding: {e}")
            return None
        
    def processing(self):
        preprocessed = self.preprocess_text()
        tokens = self.tokenize_text(preprocessed)
        padded_sequences = self.pad_sequence(tokens)
        return padded_sequences


    def get_tokens(self):
        return self.sp_model.encode_as_pieces(self.text)    
