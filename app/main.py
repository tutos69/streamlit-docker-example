import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carga del modelo y el tokenizador
model_id = "tutos69/LLM2QR1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")

# Función principal de la aplicación Streamlit
def main():
   
    st.title("Simulador de Conversaciones con IA")

    user_input = st.text_area("Ingrese su texto aquí:")

    if st.button("Generar Respuesta"):
        response = stream(user_input)
        st.text_area("Respuesta:", value=response, height=300)

def stream(user_prompt):
    runtimeFlag = "cuda:0"
    system_prompt = 'A continuación se muestra una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escriba una respuesta que complete adecuadamente la solicitud.\n\n'
    B_INST, E_INST = "### Instrucción:\n", "### Respuesta:\n"

    prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n\n{E_INST}"
    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)
    generated_ids = model.generate(**inputs,  max_new_tokens=100, max_length=60, temperature=0.7)
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return decoded_output

if __name__ == "__main__":
    main()