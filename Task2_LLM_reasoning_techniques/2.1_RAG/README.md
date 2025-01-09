# RAG steps
1. Load the documents and chunk them.
2. Get the text embedding using an embedding model.
3. Store the embedding and the corresponding text into a Database.
4. When a query comes, use the retriever to get the relevant documents.
5. Put this information into the prompts, and then generate the answer.

LightRAG需要让api model和embedding model适配它提供的接口，遂采用DIY RAG的方式。

# 参考
https://blog.csdn.net/m0_65555479/article/details/143781140
https://blog.csdn.net/Python_0011/article/details/139752344