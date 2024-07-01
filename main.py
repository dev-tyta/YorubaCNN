import torch
import streamlit as st
from your_preprocessing_module import preprocess_text  # You'll need to implement this
from your_model_module import YorubaCNN  # Your model class

# Load the trained model
def load_model(model_path):
    model = YorubaCNN(vocab_size, embed_dim, num_classes)  # Initialize with your parameters
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Perform NER
def perform_ner(model, text):
    # Preprocess the text
    preprocessed = preprocess_text(text)
    
    # Convert to tensor and get predictions
    with torch.no_grad():
        input_tensor = torch.tensor(preprocessed).unsqueeze(0)  # Add batch dimension
        predictions = model(input_tensor)
    
    # Convert predictions to entity labels
    _, predicted_labels = torch.max(predictions, dim=2)
    
    return predicted_labels.squeeze().tolist()

# Map numeric labels to entity types
label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}  # Adjust based on your labels

# Streamlit app
def main():
    st.title("Yoruba Named Entity Recognition")

    # Load the model
    model = load_model('.pth')

    # Text input
    text = st.text_area("Enter Yoruba text:", height=200)

    if st.button("Perform NER"):
        if text:
            # Perform NER
            labels = perform_ner(model, text)
            
            # Display results
            words = text.split()  # Simple tokenization, you might need a more sophisticated approach
            for word, label in zip(words, labels):
                entity_type = label_map[label]
                if entity_type != 'O':
                    st.markdown(f"**{word}** - *{entity_type}*")
                else:
                    st.write(word)
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()