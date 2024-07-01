import torch
from cnn_ner_model import YorubaCNN  
from data_handling import TextProcessing  

class YorubaNerModel:
    def __init__(self):
        self.vocab_size = 32000
        self.num_classes = 9
        self.embed_dim = 256
        self.model_path = './models/ner_model/yoruba_ner_model.pth'
        self.model = self.load_model()

    def load_model(self):
        model = torch.load(self.model_path, map_location=torch.device('cpu'))
        model.eval()
        return model

    def perform_ner(self, text):
        text_processor = TextProcessing(text)
        input_tensor = text_processor.processing()

        with torch.no_grad():
            predictions = self.model(input_tensor)

        predicted_labels = torch.argmax(predictions.squeeze(), dim=1)  # Assuming dim=1 is the classes dimension
        labels_predicted = predicted_labels.tolist()

        # Decode each label to its corresponding entity
        decoded_labels = [self.decode_label(label) for label in labels_predicted]
        return decoded_labels

    def decode_label(self, label):
        label_map = {0: 'B-DATE', 1: 'B-LOC', 2: 'B-ORG', 3: 'B-PER', 4: 'I-DATE', 5: 'I-LOC', 6: 'I-ORG', 7: 'I-PER', 8: 'O'}
        return label_map.get(label, 'Unknown')
