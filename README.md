# Retrieval-Augmented Generation (RAG)

A little toy to explore the possibilties of [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401).

The idea is to have a generative language model awnser questions exclusively on the basis of the provided sources. 

An interactive minimalist example with a web interface is provided. 

It relies on Flask, transformers, ctransformers and Llama Index.

Other modules may be needed depending on the format of the sources (PDF, DOCX, CSV, etc).

The English translation of The Little Prince in PDF format is provided as an example.

CAVEATS

- This is not a search engine. You may need a bit of trial-and-error with prompt engineering in order to get the right answers.
  
- This is not a chatbot as it does not keep any memory of the conversation threat. It is just for Q&A.

- Only plain text is correctly parsed. Images, tables and graphs will be ignored at this point.

- With the current model [tulu-2-dpo-13b.Q8_0.gguf](https://huggingface.co/TheBloke/tulu-2-dpo-13B-GGUF), you will need a GPU with, at the very least, 20GB of RAM. But you can switch to a smaller model. 
