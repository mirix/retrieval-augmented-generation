# Minimalistic chatbot app example combining LLMs and Retrieval-Augmented Generation.
# The idea is to have a generative language model answer questions exclusively on the basis 
# of the provided source material as opposed to relying on previous knowledge or hallucination.
# MORE INFO
# https://github.com/mirix/retrieval-augmented-generation

import os
n_cores = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# I am explicitly using two GPUs
# You can change this to 0 if you only have one, etc.
# You can also comment this out and check how to use device='auto'
# to autodetect the hardware configuration 
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"

### REQUIREMENTS ###

# Flask, PyMuPDF, transformers, ctransformers, llama_index
# You may want to install a version of PyTorch that matches your requirements
# (as opposed to the one pulled by transformers)
# Check the error messages for additional dependencies

from flask import Flask, render_template, request, Response

import fitz
from unidecode import unidecode
import remove_header_footer

from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from llama_index import (Document,
                         ServiceContext,
                         VectorStoreIndex,
                         SimpleKeywordTableIndex,
                         KnowledgeGraphIndex,
                         StorageContext,
                         set_global_service_context,
                         SimpleDirectoryReader)

from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts import PromptTemplate
from llama_index.tools.query_engine import QueryEngineTool

from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
)
from llama_index.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.response_synthesizers import TreeSummarize


### APP ###

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
	return render_template('index.html')

### MODEL ###
# We are using Tulu v2 7B DPO.
# We are using GGUF quantisation because it enables us to use the CPU and decide how many layers we want to offload to the GPU.
# For GPU-only AQW-quantised models will likely result in better performance (check what TheBloke is offering in Huggingface).
# With the current model, if the context is not too long, probably 8GB of VRAM will suffice. You can switch to the Q4 if you want to save VRAM.
# Downloading the models may take a while (several GB). You may prefer to download them beforehand and point to the local folders instead.

name = 'cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser'
model_name = 'kroonen/dolphin-2.6-mistral-7b-dpo-laser-GGUF'
model_file = 'dolphin-2.6-mistral-7b-dpo-laser-Q8_0.gguf'

# Set the gpu_layers according to your system. In this case, my system does not use more than 35. 0 means no GPU, CPU-only
# Llama 2 models such as Tulu, can by default handle a maximum of 4096 tokens. But there are models with longer context window out there.
model = AutoModelForCausalLM.from_pretrained(model_name, model_file=model_file, context_length=4096, gpu_layers=512, seed=0, hf=True)
											
tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)

# Current models very sensitive to the wording of the system prompt (and the prompt in general).
# Finding the right working may take some trial-and-error. Some call this "prompt engineering". 
# I call it dumb models wasting my time.

system_prompt = 'Context information from multiple sources is provided. Each source may or may not have a relevance score attached to it. Given the information from multiple sources and their associated relevance scores (if provided) and not prior knowledge, answer the question. If the answer cannot be inferred from the context, state that the question is "out of scope" and do not provide any answer'
#system_prompt = 'Please, check if the anwser can be inferred from the pieces of context provided. If the answer cannot be inferred from the context, just literally state that the question is "out of scope" and do not provide any answer.'
prompt_template = '<|im_start|>system\n' + system_prompt + '<|im_end|>\n<|im_start|>user\n{query_str}<|im_end|>\n<|im_start|>assistant\n'

llm = HuggingFaceLLM(
	model=model, 
	tokenizer=tokenizer, 
	query_wrapper_prompt=PromptTemplate(prompt_template),
	context_window=3072,
	max_new_tokens=512,
	tokenizer_kwargs={'max_length': 4096, 'legacy': False},
	# Set the gpu_layers according to your system. In this case, my system does not use more than 35. 0 means no GPU, CPU-only
	# If hyperthreading is not activated in your system you can remove the //2
	# You can set the seed for reproducibility
	model_kwargs={'n_gpu_layers': 512, 'n_threads': n_cores//2, 'seed': 0},
	# Here you can play with temperature, top_p, etc.
	# This will tune how creative the response can get. 
	# The higher the values, the more creative (and therefore hallucination-prone)
	# In this case, we prefer a very conservative response.
	generate_kwargs={'do_sample': True, 'temperature': 0.0000001, 'top_p': 0.0000001, 'top_k': 1, 'repetition_penalty': 0.9},
	)

# The embedding model can be decisive for the quality of the retrieval phase.
# If you have issues, you may want to switch to a different one.

embed_model = HuggingFaceEmbedding(model_name='thenlper/gte-large')

### CONTEX ###

# The chunk size needs to be tuned.
# The longer, the more resources the app will use.
# Rule of thumb:
# Smaller chunk size -> More information will be retreived.
# Longer chunk size -> More cwantonnections between distant pieces of information (if your model is good enough).
# Here we are using small models so we go for the smallest possible size.

service_context = ServiceContext.from_defaults(
    chunk_size=200,
    llm=llm,
    embed_model=embed_model
)

set_global_service_context(service_context)

# DOCUMENTS 

# The quality of the extracted text formatting is also crucial for the retrieval phase.
# You need to make sure you do this right, as inconsitent formatting may result in incomplete retrieval.

# For testing purposes, we are using a single PDF file.
pdf_path = 'pdf/SQE_Terms_and_Conditions.pdf'

# Define bounding box excluding headers and footers
bounding_box = fitz.Rect(remove_header_footer.remove_hf(pdf_path))

# We use PyMuPDF to extract the text from the PDF
doc = fitz.open(pdf_path)

# We extract blocks (roughly speaking paragraphs).
# Block are identified primariry from geometrical information.

paragraphs = []
for page in doc:
	blocks = page.get_text('blocks', clip=bounding_box)
	for block in blocks:
		paragraph = unidecode(block[4]).replace('\n', ' ').strip()
		paragraph = ' '.join(paragraph.split())
		# We remove info about images. You will need to adapt this.
		if not paragraph.startswith('<image: ') and not paragraph.endswith('.jpg') and paragraph != '':
			paragraphs.append(paragraph)

# PyMuPDF does the extraction page by page.
# We try to merge truncated paragraphs. 
# The criteria are the stop tokens below.
# You can resort to an NLP library rather than using an ad hoc list.
# These are obviously, language-dependent.
stops = ['.', '!', '?', ':', '"']
			
merged_paragraphs = paragraphs[:1]
for p in paragraphs[1:]:
	if merged_paragraphs[-1][-1] not in stops:
		merged_paragraphs[-1] = merged_paragraphs[-1] + ' ' + p
	else:
		merged_paragraphs.append(p)
	
text = '\n\n'.join(merged_paragraphs)

documents = [Document(text=text)]
print('Number of documents:', len(documents))

### NODES ###

nodes = service_context.node_parser.get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# INDEX

#index = VectorStoreIndex.from_documents(documents)

keyword_index = SimpleKeywordTableIndex(
    nodes,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
)

vector_index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
)

graph_index = KnowledgeGraphIndex(
	nodes, 
	storage_context=storage_context, 
	service_context=service_context, 
	show_progress=True
)

### QUERY ###

#query_engine = index.as_query_engine()

keyword_query_engine = keyword_index.as_query_engine()
vector_query_engine = vector_index.as_query_engine()
graph_query_engine = graph_index.as_query_engine()

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_query_engine
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine
)

graph_tool = QueryEngineTool.from_defaults(
    query_engine=graph_query_engine
)

tree_summarize = TreeSummarize()

query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[
        keyword_tool,
        vector_tool,
        graph_tool,
    ],
    summarizer=tree_summarize,
)

### APP FUNCTION ###

@app.route('/get')
def get_bot_response():
	
	try: msg
	except NameError: msg = ''
	
	prompt = request.args.get('msg')
	response = str(query_engine.query(prompt))
	
	if response[-1] != '.':
		response = '.'.join(response.split('.')[:-1]) + '.'
		
	if 'out of scope' in response:
		response = "Question out of scope. I can only answer questions pertaining SQE's Terms and Conditions agreement."
	
	return response

# Open the URLs indicated in the console output in your browser.
# You can choose another port.
# For instance, the following will work locally:
# http://localhost:5000
# http://127.0.0.1:5000
# If your server is accesible in a local network or the internet, you can do the following from a remote client:
# http://HOST_NAME:5000
# http://HOST_IP:5000

if __name__ == '__main__':
   app.run(host = "0.0.0.0", port = 5000, debug = False)

