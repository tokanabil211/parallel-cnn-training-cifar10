import streamlit as st
from openai import OpenAI

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# System prompt to guide the model
SYSTEM_PROMPT = (
    "You are an AI assistant that receives a sequential PyTorch training script and rewrites it into "
    "a parallel MPI version using mpi4py. Ensure each process handles a subset of the data, synchronizes "
    "gradients using MPI.Allreduce, and only rank 0 evaluates and prints results."
)

# Function to call OpenRouter model
def convert_code(code: str) -> str:
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1-distill-qwen-32b:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": code}
            ],
            extra_headers={
                "HTTP-Referer": "http://localhost",  # optional
                "X-Title": "PyTorch-to-MPI-Agent"     # optional
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Streamlit UI setup
st.set_page_config(page_title="Sequential → MPI Converter", layout="wide")
st.title("🚀 Convert Sequential PyTorch Code to Parallel (MPI)")
st.markdown("Paste your PyTorch training code or upload a file, and the AI agent will convert it using `mpi4py`.")

# Input area
code_input = st.text_area("🧾 Paste your sequential code here:", height=300)

# Optional file upload
uploaded_file = st.file_uploader("Or upload a .py file", type=["py"])
if uploaded_file:
    code_input = uploaded_file.read().decode("utf-8")

# Convert button
if st.button("⚡ Convert to MPI"):
    if not code_input.strip():
        st.warning("Please paste some code or upload a file.")
    else:
        with st.spinner("Converting code using OpenRouter + DeepSeek model..."):
            output_code = convert_code(code_input)
        st.success("✅ Conversion Complete!")
        st.text_area("📄 Converted Parallel Code:", value=output_code, height=300)
        st.download_button("💾 Download Converted Code", output_code, file_name="converted_parallel.py")
