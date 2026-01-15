import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model 
@st.cache_resource
def load_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# UI
st.title("GPT-2 Text Generator")
st.write("Enter a prompt and generate text using GPT-2")

prompt = st.text_input("Enter your prompt", "India is known for")

max_tokens = st.slider("Max new tokens", 20, 100, 50)
temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
top_p = st.slider("Top-p", 0.1, 1.0, 0.9)

if st.button("Generate Text"):
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2,
        do_sample=True
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.subheader("Generated Text")
    st.write(result)
