# RAG with Knowledge Graphs

A little toy to explore the possibilties of combining [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) with [Knowledge Graphs](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo.html).

The idea is to have a generative language model to awnser questions exclusively on the basis of the provided sources. 

An interactive minimalist example with a web interface is provided. 

It relies on Flask, transformers, ctransformers and Llama Index.

Other modules may be needed depending on the format of the sources (PDF, DOCX, CSV, etc).

The English translation of The Little Prince in PDF format is provided as an example.

CAVEATS

- This is not a search engine. You may need a bit of trial-and-error prompt engeneering in order to get the right answers.
  
- This is not a chatbot as it does not keep any memory of the conversatio threat. It is just for question/answer.

- Only plain text is correctly parsed. Images, tables and graphs will be ignored at this point.

- Generating the Knowledge graphs is very time consuming. Be patient. In theory, the graphs can be saved and reloaded (not tested). As far as the app is up, they are stored in memory.

- With the current model [solar-10.7b-instruct-v1.0.Q6_K.gguf](https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF), you will need a GPU with, at the very least, 16GB of RAM. But you can switch to a smaller model. 
