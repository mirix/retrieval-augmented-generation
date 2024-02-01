# Retrieval-Augmented Generation (RAG)

A little toy to explore the possibilities of [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401).

The idea is to have a generative language model awnser questions exclusively on the basis of the provided sources. 

An interactive minimalist example with a web interface is provided. 

It relies on Flask, transformers and Llama Index.

NOTE: I am not going to keep updating this experimental repo because the RAG protocol is now being integrated into a more complex pipeline. So everything here should be considered outdated and is provided for illustrative purposes only.


A few different scripts are provided:

1. rag_llama_index_AWQ_multi_app uses all PDF documents within a given folder as source material and [dolphin-2.6-mistral-7B-dpo-AWQ](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-AWQ) as language model. This approach is much faster and more efficient in every aspect (retrieval, reranking, etc.) than the previous ones. Recommended.
  
2. rag_llama_index_AWQ_mini_app is an older version reading one specific PDF.

3. rag_llama_index_tulu_app uses the English translation of The Little Prince in PDF format as source material [tulu-2-dpo-7b.Q5_K_M.gguf](https://huggingface.co/TheBloke/tulu-2-dpo-7B-GGUF) as language model. This is an early approach using GGUF and it is slow and inefficient. I keep it here for reference.

4. The exllamav2 script is highly experimental work in progress. It relies on [exl2-for-all](https://github.com/chu-tianxiang/exl2-for-all), but this library does not work out of the box for RAG and needs to be hacked. Working, but currently only on one GPU and slower than AWQ. 


CAVEATS

- This is not a search engine. You may need a bit of trial-and-error with prompt engineering in order to get the right answers.
  
- This is not a chatbot as it does not keep any memory of the conversation threat. It is just for Q&A.

- Only plain text is correctly parsed. Images, tables and graphs will be ignored at this point.

- The preprocessing function has been tested on a small set of PDF files with roughly similar format. It is heuristic but most likely it will not work for every file.


WHY AN UNALIGNED MODEL?

Check [this](https://www.linkedin.com/posts/ed-moman-3b4632294_are-censoredaligned-language-models-intrinsically-activity-7157769140450062336-696M?utm_source=share&utm_medium=member_desktop) as well as the discussion [here](https://www.linkedin.com/posts/carsten-draschner_what-happens-when-you-break-llm-alignment-activity-7157765084734214144-2yGt?utm_source=share&utm_medium=member_desktop). I am Ed Moman and would love to hear your opinion on this matter if you have one.

