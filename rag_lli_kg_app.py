import os
n_cores = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"

from flask import Flask, render_template, request

from llama_index import (SimpleDirectoryReader,
                         LLMPredictor,
                         ServiceContext,
                         KnowledgeGraphIndex)
                       
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding

# Folder containing the context documents
# Additional modules may need to be installed depending on the format

data_folder = 'pdf'

### APP ###

# An example prompt that works
# prompt = "What are leverage ratios?"

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
	return render_template('index.html')

### MODEL ###

name = 'upstage/SOLAR-10.7B-Instruct-v1.0'

model = AutoModelForCausalLM.from_pretrained('TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF', model_file='solar-10.7b-instruct-v1.0.Q6_K.gguf', model_type='llama',  
											context_length=4096, gpu_layers=200, hf=True)
											
tokenizer = AutoTokenizer.from_pretrained(name)

llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)

embed_model = HuggingFaceEmbedding(model_name='thenlper/gte-large')

### CONTEX ##

# Documents 

documents = SimpleDirectoryReader(data_folder).load_data()

print('Number of pages:', len(documents))

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

# Store

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Knowlege Graph Index

index = KnowledgeGraphIndex.from_documents(documents=documents,
                                           max_triplets_per_chunk=3,
                                           service_context=service_context,
                                           storage_context=storage_context,
                                           include_embeddings=True)


### QUERY ###

query_engine = index.as_query_engine(include_text=True,
                                     response_mode ="tree_summarize",
                                     embedding_mode="hybrid",
                                     similarity_top_k=5,)

### APP FUNCTION ###

@app.route('/get')
def get_bot_response():
	
	try: msg
	except NameError: msg = ''
	
	query = request.args.get('msg')
	
	prompt_template =f"""<s>[INST] Please, check if the anwser to Question below can be inferred from the pieces of context provided. 
	If the answer cannot be inferred from the context, state that the question is out of scope.

	Question: {query}
	[/INST]"""
	
	response = query_engine.query(prompt_template)
	print(response)
	
	return str(response)
	
if __name__ == '__main__':
   app.run(host = "0.0.0.0", port = 5000, debug = False)

