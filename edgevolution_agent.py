import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    model_name = "meta-llama/Llama-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

def preprocess_repository():
    # Example: Index repository content
    repo_index = {}
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith((".md", ".py", ".txt")):
                with open(os.path.join(root, file), "r") as f:
                    content = f.read()
                    repo_index[file] = content
    return repo_index

def generate_response(query, repo_index, model, tokenizer):
    # Find relevant files (basic keyword search)
    relevant_context = "\n\n".join([
        f"{file}:\n{content[:500]}" for file, content in repo_index.items() if query in content
    ])
    prompt = f"Repository Context:\n{relevant_context}\n\nUser Query: {query}"
    
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(inputs["input_ids"], max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    repo_index = preprocess_repository()

    user_query = input("Ask a question about the repository: ")
    response = generate_response(user_query, repo_index, model, tokenizer)
    print(response)
