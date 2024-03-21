import os
n_cores = os.cpu_count()//2
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# Adjust according to the number of GPUs you wish to use
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import glob
import fitz
import pandas as pd
from unidecode import unidecode
# Custom function found in the same repository
# It works for the Terms and Condition sample 
# but it has not been thoroughly tested
import remove_header_footer

#from transformers import AutoModelForCausalLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('TomGrc/FusionNet_7Bx2_MoE_v0.1')
Settings.tokenizer = tokenizer
#.encode
#max_length=8192

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.core.prompts import PromptTemplate

from llama_index.core import (
	Document,
	SimpleDirectoryReader,
	ServiceContext,
	StorageContext,
	VectorStoreIndex,
	#SimpleKeywordTableIndex,
)

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import (
	BaseRetriever,
)

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
#from llama_index.core.evaluation import FaithfulnessEvaluator

from typing import List, Optional, Sequence
from llama_index.core.base.llms.types import ChatMessage, MessageRole

### MODEL ###

# This particular model appears to work particularly well for this task
# better than the top ones on the leaderboard at the time of this writing.
model_name = '/home/emoman/Work/sqeagle/models/truthful_dpo_tomgrc_fusionnet_7bx2_moe_13b.Q4_K_M.gguf'
#model_name = 'TheBloke/dolphin-2.6-mistral-7B-dpo-laser-AWQ'
#model_name = 'solidrust/fc-dolphin-2.6-mistral-7b-dpo-laser-AWQ'
# Switiching to the large embedding model does not make a big difference
#emb_model = 'BAAI/bge-large-en-v1.5'
#emb_model = 'BAAI/bge-base-en-v1.5'
emb_model = 'mixedbread-ai/mxbai-embed-large-v1'
# Switiching to the large reraking model does not make a big difference
#rerank_model = 'BAAI/bge-reranker-large'
#rerank_model = 'BAAI/bge-reranker-base'
rerank_model = 'mixedbread-ai/mxbai-rerank-large-v1'

#model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:1', low_cpu_mem_usage=True)
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')									

# Change the prompt template according to the model.
# Prompt engineering is crucial. Try different wordings.
# The shorter, the lower the risk of hallucination.

B_SYS = "<|im_start|>system\n"
B_USER = "<|im_start|>user\n"
B_ASSISTANT = "<|im_start|>assistant\n"
END = "<|im_end|>\n"
DEFAULT_SYSTEM_PROMPT = ('Answer the question solely on the basis of the provided context information. '
						'If the answer cannot be inferred from the provided context information, state that the question is out of scope.'
)

def messages_to_prompt(
    messages: Sequence[ChatMessage], system_prompt: Optional[str] = None) -> str:
    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    string_messages.append(f"{B_SYS}{system_message_str.strip()} {END}")

    for message in messages:
        role = message.role
        content = message.content

        if role == MessageRole.USER:
            string_messages.append(f"{B_USER}{user_message.content} {END}")
        elif role == MessageRole.ASSISTANT:
            string_messages.append(f"{B_ASSISTANT}{assistant_message.content} {END}")

    string_messages.append(f"{B_ASSISTANT}")

    return "".join(string_messages)


def completion_to_prompt(completion: str, system_prompt: Optional[str] = None) -> str:
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{B_SYS}{system_prompt_str.strip()} {END}"
        f"{B_USER}{completion.strip()} {END}"
        f"{B_ASSISTANT}"
    )

'''
llm = HuggingFaceLLM(
	model=model, 
	tokenizer=tokenizer, 
	#query_wrapper_prompt=PromptTemplate(prompt_template),
	# The max number of tokens for this model is 4096
	# We leave some room for the prompt template
	context_window=8192,
	# Max number of tokens the output can contain
	max_new_tokens=512,
	#tokenizer_kwargs={'legacy': False, 'trust_remote_code': True},
	# 'max_length': 4096, 
	#model_kwargs={'n_threads': n_cores, 'seed': 0, 'use_flash_attention_2': True},
	# Extremely conservative options to avoid unecessary verbosity, repetion and hallucination
	generate_kwargs={'do_sample': True, 'prompt_lookup_num_tokens': 10, 'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.eos_token_id,
					#'temperature': 0.8, 'repetition_penalty': 1.1, 'do_sample': True}
					'temperature': 0.0000001, 'top_p': 0.0000001, 'top_k': 1, 'repetition_penalty': 0.7}
	#'prompt_lookup_num_tokens': 10, 'pad_token_id': tokenizer.eos_token_id}
	)
'''

llm = LlamaCPP(
    #model_url=model_url,
    model_path=model_name,
    #temperature=0.0000001,
    max_new_tokens=512,
    #n_threads=n_cores,
    #n_gpu_layers=200,
    context_window=8192,
    #tokenizer_kwargs={'max_length': 4096, 'legacy': False},
    model_kwargs={'n_threads': n_cores, 'seed': 0, 'use_flash_attention_2': True, 'n_gpu_layers': 200},
	#generate_kwargs={'do_sample': True, 'prompt_lookup_num_tokens': 10, 'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.eos_token_id,
	#				'temperature': 0.0000001, 'top_p': 0.0000001, 'top_k': 1, 'repetition_penalty': 0.7},
					#'temperature': 0.8, 'repetition_penalty': 1.1, 'do_sample': True}
	generate_kwargs={},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
)


### DOCUMENTS ###

#pdfs = glob.glob('documents/*.pdf')
pdfs = 'pdfs'
df = pd.read_csv('sqe_all_links.csv')

to_remove = ['terms_and_conditions_bank_europe_en_01012022', 'general_terms_and_conditions_sqbe_en_27102022']
df = df[~df['file_name'].str.contains('|'.join(to_remove))].copy()

documents = []
def txt_to_nodes(row):
	
	path = row['file_name']
	url = row['url']
	
	ext = path.split('.')[-1]
	pdf_path = pdfs + '/' + path
	title = '_'.join(path.split('.')[0].split('_')[:-1])
	
	metadata={'title': title, 'url': url}
	
	if ext == 'pdf':		
		try:		
			bounding_box = fitz.Rect(remove_header_footer.remove_hf(pdf_path))
		except Exception:
			bounding_box=None
		
		doc = fitz.open(pdf_path)
		
		# Extract text blocks (paragraphs for each page)
		paragraphs = []
		for page in doc:
			if bounding_box is not None:
				blocks = page.get_text('blocks', clip=bounding_box)
			else:
				blocks = page.get_text('blocks')
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
		
	elif ext == 'txt':
		txt_file = open(pdf_path, 'r')
		text = txt_file.read()
		txt_file.close()
	
	exclude_op = 'Andrew Hallam'
	
	if exclude_op not in text:
		documents.append(Document(text=text, metadata=metadata))
	
df.apply(lambda row: txt_to_nodes(row), axis=1)

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
	chunk_size=512,
	chunk_overlap=64,
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

top_k = 3
top_n = 6

# We will retrieve four relevant chunks with the vector retriever.
vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
# We are not using the keyword retriever any more.
#keyword_retriever = keyword_index.as_retriever(similarity_top_k=4)
# We will retrieve four relevant chunks with the old BM25 algorithm.
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

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
reranker = SentenceTransformerRerank(top_n=top_n, model=rerank_model)

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
#evaluator = FaithfulnessEvaluator(service_context=service_context)
#evaluator = FaithfulnessEvaluator(llm=llm)

### APP FUNCTION ###

def get_bot_response(prompt):
	
	response = query_engine.query(prompt)
	
	# Evaluate the relevance of the response with respect to the source material
	# by using the FaithfulnessEvaluator.
	#eval_result = evaluator.evaluate_response(response=response)
	
	# If the response does not pass the test, provide a standard response.
	# If it passes, provide the model's response.
	#print(eval_result.passing)
	#print(eval_result.score)
	#if str(eval_result.passing) == 'False':
	if response.source_nodes[0].score < 0.25:
		response_text = 'I cannot find an answer to that question in en.swissquote.lu. Please, rephrase the question.'
		source = 'No sources'
	else:
		response_text = str(response)
	
		# If the last sentence is unfinished, remove it.
		if response_text[-1] != '.':
			response_text = '.'.join(response_text.split('.')[:-1]) + '.'
		
		mark_list = ['#### SOURCES  \n']
		
		uniq_nodes = list({v.node.get_text():v for v in response.source_nodes}.values())
		
		max_range = len(uniq_nodes)
		if max_range >= 3:
			max_range = 3
		else:
			max_range = max_range
		
		for i in range(max_range):
			node = uniq_nodes[i]
			source_title = '#### ' + node.metadata['title'].replace('_', ' ').title()
			source_score = 'Relevance Score: ' + str(round(node.score, 2))
			hyperurl = '[' + node.metadata['url'] + ']' + '(' + node.metadata['url'] + ')'
			source_txt = node.node.get_text() 
			sep = '&nbsp;  \n'
			mark_list.extend([source_title, source_score, hyperurl, source_txt, sep])
		
		source = '  \n'.join(mark_list)
	
	return response_text, source

