import os
n_cores = os.cpu_count()//2
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# Adjust according to the number of GPUs you wish to use
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

from flask import Flask, render_template, request, Response

import glob
import fitz
from unidecode import unidecode
# Custom function found in the same repository
# It works for the Terms and Condition sample 
# but it has not been thoroughly tested
import remove_header_footer

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts import PromptTemplate

from llama_index import (
	Document,
	SimpleDirectoryReader,
	ServiceContext,
	StorageContext,
	VectorStoreIndex,
	#SimpleKeywordTableIndex,
)

from llama_index.retrievers import (
	BM25Retriever,
	BaseRetriever,
)

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.evaluation import FaithfulnessEvaluator

### APP ###

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
	return render_template('index.html')

### MODEL ###

# This particular model appears to work particularly well for this task
# better than the top ones on the leaderboard at the time of this writing.
#model_name = 'TheBloke/WestLake-7B-v2-AWQ'
model_name = 'TheBloke/dolphin-2.6-mistral-7B-dpo-laser-AWQ'
# Switiching to the large embedding model does not make a big difference
#emb_model = 'BAAI/bge-large-en-v1.5'
emb_model = 'BAAI/bge-base-en-v1.5'
# Switiching to the large reraking model does not make a big difference
#rerank_model = 'BAAI/bge-reranker-large'
rerank_model = 'BAAI/bge-reranker-base'

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:1', low_cpu_mem_usage=True)									
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

# Change the prompt template according to the model.
# Prompt engineering is crucial. Try different wordings.
# The shorter, the lower the risk of hallucination.
system_prompt = ('Answer the question solely on the basis of the provided context information. '
                 'If the answer cannot be inferred from the provided context information, state that the question is out of scope.'
                )

prompt_template = '<|im_start|>system\n' + system_prompt + '<|im_end|>\n<|im_start|>user\n{query_str}<|im_end|>\n<|im_start|>assistant\n'

llm = HuggingFaceLLM(
	model=model, 
	tokenizer=tokenizer, 
	query_wrapper_prompt=PromptTemplate(prompt_template),
	# The max number of tokens for this model is 4096
	# We leave some room for the prompt template
	context_window=3072,
	# Max number of tokens the output can contain
	max_new_tokens=512,
	tokenizer_kwargs={'max_length': 4096, 'legacy': False},
	model_kwargs={'n_threads': n_cores, 'seed': 0},
	# Extremely conservative options to avoid unecessary verbosity, repetion and hallucination
	generate_kwargs={'do_sample': True, 'temperature': 0.0000001, 'top_p': 0.0000001, 'top_k': 1, 'repetition_penalty': 0.7, 
	'prompt_lookup_num_tokens': 10, 'pad_token_id': tokenizer.eos_token_id}
	)

### DOCUMENTS ###

pdfs = glob.glob('documents/*.pdf')

documents = []
for pdf_path in pdfs:
	# Calculate the cordinates of a box containing the text
	# but excluding the header and the footer.
	# This uses a custom function provided in the same repo.
	# Not well tested.
	bounding_box = fitz.Rect(remove_header_footer.remove_hf(pdf_path))
	
	doc = fitz.open(pdf_path)
	
	# Extract text blocks (paragraphs for each page)
	paragraphs = []
	for page in doc:
		blocks = page.get_text('blocks', clip=bounding_box)
		for block in blocks:
			paragraph = unidecode(block[4]).replace('\n', ' ').strip()
			paragraph = ' '.join(paragraph.split())
			if not paragraph.startswith('<image: ') and not paragraph.endswith('.jpg') and paragraph != '':
				paragraphs.append(paragraph)
	
	# Attempt to merge paragraphs that have been split
	# in different pages or columns
	stops = ['.', '!', '?', ':', '"']			
	merged_paragraphs = paragraphs[:1]
	for p in paragraphs[1:]:
		if merged_paragraphs[-1][-1] not in stops:
			merged_paragraphs[-1] = merged_paragraphs[-1] + ' ' + p
		else:
			merged_paragraphs.append(p)
		
	text = '\n\n'.join(merged_paragraphs)
	text = text.replace('k d b d d ffi d d h b', '')
	
	documents.append(Document(text=text))

# Build a Llama index document from the extracted text
print('Number of documents:', len(documents))

### QUERY RETRIEVER ###

# Split the document in chunks and compute the embeddings for each chunk
# The chuck size is absolutely crucial:
# There is a trade-off between retrieval exhaustivity and context size.
# The smaller the chunck size, the more exhaustive the retrieval phase,
# but this fragmentation provides less context to the model, 
# thus making "connecting the dots" more difficult.
# Here we are using very small chuncks to favour retrieval exhaustivity.
# The chunk overlap is also important.
# You may wish to try several combinations of these values.
service_context = ServiceContext.from_defaults(
	chunk_size=200,
	chunk_overlap=40,
	llm=llm,
	embed_model=HuggingFaceEmbedding(model_name=emb_model)
)

# Create nodes from the chunks.
nodes = service_context.node_parser.get_nodes_from_documents(documents)

# Build the storate.
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# Build the vector index.
vector_index = VectorStoreIndex(
	nodes,
	storage_context=storage_context,
	service_context=service_context,
	show_progress=True,
)

'''
# Build the keyword index.
# We have commented this out as it is time consumming and BM25 seems superior
keyword_index = SimpleKeywordTableIndex(
	nodes,
	storage_context=storage_context,
	service_context=service_context,
	show_progress=True,
)
'''

# We will retrieve four relevant chunks with the vector retriever.
vector_retriever = vector_index.as_retriever(similarity_top_k=5)
# We are not using the keyword retriever any more.
#keyword_retriever = keyword_index.as_retriever(similarity_top_k=4)
# We will retrieve four relevant chunks with the old BM25 algorithm.
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

# We build a hybrid retriever by combining the vector retreiver and the BM25 retriever.
# This approach seems both fast and accurate.
class HybridRetriever(BaseRetriever):
	#def __init__(self, vector_retriever, keyword_retriever, bm25_retriever):
	def __init__(self, vector_retriever, bm25_retriever):
		self.vector_retriever = vector_retriever
		#self.keyword_retriever = keyword_retriever
		self.bm25_retriever = bm25_retriever
		super().__init__()

	def _retrieve(self, query, **kwargs):
		vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
		#keyword_nodes = self.keyword_retriever.retrieve(query, **kwargs)
		bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)

		all_nodes = []
		node_ids = set()
		#for n in bm25_nodes + vector_nodes + keyword_nodes:
		for n in bm25_nodes + vector_nodes:
			if n.node.node_id not in node_ids:
				all_nodes.append(n)
				node_ids.add(n.node.node_id)
		return all_nodes

#hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever, bm25_retriever)
hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
# We rerank the combined results provided by the hybrid retriever
reranker = SentenceTransformerRerank(top_n=10, model=rerank_model)

# We set the query engine with the hybrid retriever and the reranker.
query_engine = RetrieverQueryEngine.from_args(
	retriever=hybrid_retriever,
	node_postprocessors=[reranker],
	service_context=service_context,
	storage_context=storage_context,
	response_mode='tree_summarize',
)

# To further attempt to prevent hallucination we will check whether or not the response
# seem relevant as compared to the source material.
evaluator = FaithfulnessEvaluator(service_context=service_context)

### APP FUNCTION ###

@app.route('/get')
def get_bot_response():
	
	try: msg
	except NameError: msg = ''
	
	# Run the query by merging the user prompt into the prompt template
	prompt = request.args.get('msg')
	response = query_engine.query(prompt)
	
	# Evaluate the relevance of the response with respect to the source material
	# by using the FaithfulnessEvaluator.
	eval_result = evaluator.evaluate_response(response=response)
	
	# If the response does not pass the test, provide a standard response.
	# If it passes, provide the model's response.
	if str(eval_result.passing) == 'False':
		response = 'I am under the impression that the question is unrelated to our Legal Documents. If you think I am wrong, please, rephrase the question.'
	else:
		response = str(response)
	
	# If the last sentence is unfinished, remove it.
	if response[-1] != '.':
		response = '.'.join(response.split('.')[:-1]) + '.'
	
	return response

# You can set the port here. Check the Flask documentation for a range of valid ports.
# You can access your web app by CRTL + click on the URL provided on the console.
if __name__ == '__main__':
   app.run(host = "10.56.88.201", port = 5000, debug = False)
