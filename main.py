import torch
import streamlit as st
from src.model_loading import YorubaNerModel



ner_model = YorubaNerModel()


def main():
    st.title("Yoruba Named Entity Recognition")

    # Text input
    text = st.text_area("Enter Yoruba Articles:", height=200)

    if st.button("Perform NER"):
        if text:
            labels = ner_model.perform_ner(text)
            if labels:
                entities = ner_model.decode_label(labels)
                for entity, entity_type in entities:
                    st.write(f"{entity}: {entity_type}")
            else:
                st.write("No entities found.")
        else:
            st.write("Please enter some text.")


if __name__ == "__main__":
    main()