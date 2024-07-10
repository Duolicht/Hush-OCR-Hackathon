import streamlit as st
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor
from PIL import Image
import warnings
import io
warnings.filterwarnings("ignore")

st.title("OCR-To-Json")

@st.cache_resource
def load_model():
    model_id = "YOUR_MODEL_ID_HERE"
    bnb_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=True,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map="cuda",
                                                 trust_remote_code=True,
                                                 torch_dtype="auto",
                                                 _attn_implementation="eager",
                                                 quantization_config=bnb_config)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

model, processor = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Chat input
if prompt := st.chat_input("What would you like to know about the receipt?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if uploaded_file is None:
            st.warning("Please upload an image before sending a message.")
            st.stop()
        with st.spinner("Analyzing receipt..."):
          messages = [
              {"role": "user", "content": f"<|image_1|>\nYou are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object. Don't make up value not in the Input. Include every information present in the receipt. Output must be a well-formed JSON object.```json."}
          ]

          prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

          inputs = processor(prompt, [image], return_tensors="pt").to("cuda")
          generation_args = {
              "max_new_tokens": 1024,
              "temperature": 0.0,
              "do_sample": False,
          }
          generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
          # remove input tokens
          generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
          response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("```")[0]

          st.markdown(f"```json\n{response} ```")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
