# Retrieval-Augmented Generation (RAG)

A little toy to explore the possibilties of [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401).

The idea is to have a generative language model awnser questions exclusively on the basis of the provided sources. 

An interactive minimalist example with a web interface is provided. 

It relies on Flask, transformers and Llama Index.

Two different scripts are provided:

1. rag_llama_index_tulu_app uses the English translation of The Little Prince in PDF format as source material [tulu-2-dpo-7b.Q5_K_M.gguf](https://huggingface.co/TheBloke/tulu-2-dpo-7B-GGUF) as language model. This is an early approach and it is slow and inefficient. I keep it here for reference.

2. rag_llama_index_AWQ_tiny_app uses SQE's Terms and Conditions as source material and [dolphin-2.6-mistral-7B-dpo-AWQ](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-AWQ) as language model. This approach is much faster and more efficient in every aspect (retrieval, reranking, etc.). 



CAVEATS

- This is not a search engine. You may need a bit of trial-and-error with prompt engineering in order to get the right answers.
  
- This is not a chatbot as it does not keep any memory of the conversation threat. It is just for Q&A.

- Only plain text is correctly parsed. Images, tables and graphs will be ignored at this point.

- With the current GUFF model you will need a GPU with, at the very least, 8GB of RAM. But you can switch to a smaller GGUF model. The AWQ model is smaller.
