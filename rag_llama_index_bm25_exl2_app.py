import os
n_cores = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"

from flask import Flask, render_template, request, Response

import fitz
from unidecode import unidecode
import remove_header_footer

#from ctransformers import AutoModelForCausalLM
from model import Exl2ForCausalLM
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

model_name = 'bartowski/dolphin-2.6-mistral-7b-dpo-laser-exl2'
revision = '4_0'
#model_file = 'dolphin-2.6-mistral-7b-dpo-laser-Q8_0.gguf'
emb_model = 'BAAI/bge-large-en-v1.5'
#rerank_model = 'BAAI/bge-reranker-large'
rerank_model = 'BAAI/bge-reranker-base'

model = Exl2ForCausalLM.from_quantized(model_name, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, legacy=False)

system_prompt = ('Answer the question solely on the basis of the provided context information. '
                 'If the answer cannot be inferred from the provided context information, state that the question is out of scope.'
                )

prompt_template = '<|im_start|>system\n' + system_prompt + '<|im_end|>\n<|im_start|>user\n{query_str}<|im_end|>\n<|im_start|>assistant\n'

llm = HuggingFaceLLM(
	model=model, 
	tokenizer=tokenizer, 
	query_wrapper_prompt=PromptTemplate(prompt_template),
	context_window=3072,
	max_new_tokens=256,
	tokenizer_kwargs={'max_length': 4096, 'legacy': False},
	model_kwargs={'n_threads': n_cores, 'seed': 0},
	generate_kwargs={'do_sample': True, 'temperature': 0.0000001, 'top_p': 0.0000001, 'top_k': 1, 'repetition_penalty': 0.9},
	)

### DOCUMENTS ###

pdf_path = 'pdf/SQE_Terms_and_Conditions.pdf'

bounding_box = fitz.Rect(remove_header_footer.remove_hf(pdf_path))

doc = fitz.open(pdf_path)

paragraphs = []
for page in doc:
	blocks = page.get_text('blocks', clip=bounding_box)
	for block in blocks:
		paragraph = unidecode(block[4]).replace('\n', ' ').strip()
		paragraph = ' '.join(paragraph.split())
		if not paragraph.startswith('<image: ') and not paragraph.endswith('.jpg') and paragraph != '':
			paragraphs.append(paragraph)

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

### QUERY RETRIEVER ###

service_context = ServiceContext.from_defaults(
	chunk_size=200,
	chunk_overlap=40,
	llm=llm,
	embed_model=HuggingFaceEmbedding(model_name=emb_model)
)

nodes = service_context.node_parser.get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

vector_index = VectorStoreIndex(
	nodes,
	storage_context=storage_context,
	service_context=service_context,
	show_progress=True,
)

'''
keyword_index = SimpleKeywordTableIndex(
	nodes,
	storage_context=storage_context,
	service_context=service_context,
	show_progress=True,
)
'''

vector_retriever = vector_index.as_retriever(similarity_top_k=4)
#keyword_retriever = keyword_index.as_retriever(similarity_top_k=9)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)

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
reranker = SentenceTransformerRerank(top_n=8, model=rerank_model)

query_engine = RetrieverQueryEngine.from_args(
	retriever=hybrid_retriever,
	node_postprocessors=[reranker],
	service_context=service_context,
	storage_context=storage_context,
	response_mode='tree_summarize',
)

evaluator = FaithfulnessEvaluator(service_context=service_context)

### APP FUNCTION ###

@app.route('/get')
def get_bot_response():
	
	try: msg
	except NameError: msg = ''
	
	prompt = request.args.get('msg')
	response = query_engine.query(prompt)
	
	eval_result = evaluator.evaluate_response(response=response)
	
	if str(eval_result.passing) == 'False':
		response = 'I am under the impression that the question is unrelated to our Terms and Conditions. If you think I am wrong, please, rephrase the question.'
	else:
		response = str(response)
		
	if response[-1] != '.':
		response = '.'.join(response.split('.')[:-1]) + '.'
	
	return response

if __name__ == '__main__':
   app.run(host = "0.0.0.0", port = 5000, debug = False)
