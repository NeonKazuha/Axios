import streamlit as st
import torch
import torch.nn.functional as F

# Import your model and necessary components
from axiosthingy.py import BigramLanguageModel, device, block_size, Preprocess

# Load your trained model
@st.cache_resource
def load_model():
    model = BigramLanguageModel()
    model.load_state_dict(torch.load('avengers.pth'), map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

text_preprocessor = Preprocess()

st.title('Avengers Story Generator')

# User input
input_text = st.text_area("Enter some starting text:", "Once upon a time")

# Number of tokens to generate
max_new_tokens = st.slider("Number of tokens to generate:", 1, 500, 100)

# Generate button
if st.button('Generate Text'):
    context = torch.tensor(text_preprocessor.encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)
    
    generated_text = text_preprocessor.decode(generated_tokens[0].tolist())
    
    st.write("Generated Text:")
    st.write(generated_text)

st.sidebar.header("Model Information")
st.sidebar.write(f"Vocabulary Size: {model.token_embedding_table.num_embeddings}")
st.sidebar.write(f"Embedding Dimension: {model.token_embedding_table.embedding_dim}")
st.sidebar.write(f"Number of Layers: {len(model.blocks)}")
st.sidebar.write(f"Window Size: {model.blocks[0].sa.heads[0].window_size}")