import torch
import streamlit as st
from src.model_loading import YorubaNerModel
from src.data_handling import TextProcessing



ner_model = YorubaNerModel()

# Streamlit app
def main():
    st.title("Yoruba Named Entity Recognition")

    # Text input
    text = st.text_area("Enter Yoruba Articles:", height=200)

    if st.button("Perform NER"):
        if text:
            labels = ner_model.perform_ner(text)
            if labels:
                entities = ner_model.get_entity_names(text, labels)
                for entity, entity_type in entities:
                    print(f"{entity}: {entity_type}")
            
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