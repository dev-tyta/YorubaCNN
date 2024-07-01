import torch
from data_handling import TextProcessing
from cnn_ner_model import YorubaCNN


class YorubaNerModel:
    def __init__(self):
        self.vocab_size = 32000
        self.num_classes = 9
        self.embed_dim = 256
        self.model_path = './models/ner_model/yoruba_ner_model.pth'
        self.model = self.load_model()


    def load_model(self):
        try:
            # Try loading the entire model
            model = torch.load(self.model_path)
        except AttributeError:
            # If that fails, try loading just the state dict
            model = YorubaCNN(self.vocab_size, self.embed_dim, self.num_classes)  # Initialize with your parameters
            model.load_state_dict(torch.load(self.model_path))
        
        model.eval()
        return model


    def perform_ner(self, text):
        if self.model is None:
            print("Model not loaded. Cannot perform NER.")
            return None

        try:
            # Preprocess the text
            self.text_processor = TextProcessing(text)
            preprocessed = self.text_processor.processing()

            if preprocessed is None:
                return None

            print(f"Preprocessed shape: {preprocessed.shape}")

            # Ensure preprocessed is 2D: [batch_size, sequence_length]
            if preprocessed.dim() == 1:
                preprocessed = preprocessed.unsqueeze(0)
            elif preprocessed.dim() == 3:
                preprocessed = preprocessed.squeeze(-1)

            print(f"After adjustment shape: {preprocessed.shape}")

            # No need to add batch dimension, it's already there
            input_tensor = preprocessed

            # Get predictions
            with torch.no_grad():
                predictions = self.model(input_tensor)

            print(f"Predictions shape: {predictions.shape}")

            # Convert predictions to entity labels
            _, predicted_labels = torch.max(predictions, dim=1)

            labels_predicted = predicted_labels.squeeze().tolist()
            print(f"Predicted Labels: {labels_predicted}")
            return labels_pridcted
        except Exception as e:
            print(f"Error performing NER: {e}")
            return None


    def get_entity_names(self, text, labels):
        
        if self.text_processor is None:
            print("Text not processed. Cannot get entity names.")
            return None

        try:
            
            tokens = self.text_processor.get_tokens()
            entity_names = []
            for token, label in zip(tokens, labels):
                if label != 0:  # Assuming 0 is for non-entity
                    entity_names.append((token, self.get_label_name(label)))
            return entity_names
        except Exception as e:
            print(f"Error getting entity names: {e}")
            return None


    def get_label_name(self, label):
        # Define your label mapping here
        label_map = {'B-DATE': 0, 'B-LOC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-DATE': 4, 'I-LOC': 5, 'I-ORG': 6, 'I-PER': 7, 'O': 8}
        return label_map.get(label, 'Unknown')


ner_model = YorubaNerModel()
text = "Eyi ni apẹẹrẹ gbolohun ni ede Yoruba."
labels = ner_model.perform_ner(text)
if labels:
    entities = ner_model.get_entity_names(text, labels)
    for entity, entity_type in entities:
        print(f"{entity}: {entity_type}")