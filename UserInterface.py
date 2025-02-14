from data_loader import Loader
from chroma import Chroma
from embeddings import embedding
from llms import LLM

ChromaInstance = Chroma()
LoaderInstance = Loader(chroma_instance=ChromaInstance)

# change it with your text file or pdf file path.
example_path = "use-cases/local_llm_rag/example.txt"

LoaderInstance.load_document(example_path)

# Use in True whether you wan't that the model uses less vram but you need a GPU.
# Use in False whether you want to use it fully fp precision (will use more vram)
quantized = True

llm = LLM(default_model="timpal0l/mdeberta-v3-base-squad2", tasks=["question-answering"], initial_config={
    "device_map":"auto"
})

while True:
    user_query = input("- Prompt: ")
    context = ChromaInstance.query(embedding(user_query), {})
    
    user_query_template = create_template(context, user_query)
  
    completion = llm.task("question-answering", user_query_template)

    print(completion)