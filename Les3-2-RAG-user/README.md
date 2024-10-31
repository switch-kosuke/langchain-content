# RAG user
本パートでは、ユーザーの質問や入力を受け取り、それらを埋め込む事で、ユーザーが何を望んでいるのかのベクトル表現を得る.  
そして、ベクトルストアから質問ベクトルの近くにある関連ベクトルを取得する.  
そのあとに、それらの書類を取り出し、チャンクを補強してLLMに返す処理をする.  

LangChainはこれらの処理をたった一行でしてくれる. これがLangChainの強みである.  

## Code
```python
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrival_chain = create_retrieval_chain(
    retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
)

result = retrival_chain.invoke(input={"input": query})

```
- プロンプト  
	- 「langchain-ai/retrieval-qa-chat」  
		```txt
		Answer any use questions based solely on the context below:

		<context>
		{context}
		</context>
		```
		Contextには、人間の入力やチャットヒストリーが含まれる.  
		このプロンプトをLLMに送ると、LangChainはセマンティック検索でベクトルストアから得た関連ドキュメントをコンテキストに入力する.  
		
		LLMは統計的な推論なので、文脈に沿った回答のみでプロンプトを水増しする方法は必ずしも役に立たない.    

	- custom  
		```python
		rag_chain = (
			{"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
			| custom_rag_prompt
			| llm
		)
		```

		- [RunnablePassthrough](https://python.langchain.com/v0.1/docs/expression_language/primitives/passthrough/)  
			ここでは、questionキーに引数として、質問を渡す事を意味している.  
			そのため、質問の値には最終的なLLMの呼び出しに反映される.  

		- vectorstore.as_retriever() | format_docs  
			ベクターストアから質問に関連する書類を受け取り、それらをフォーマットドキュメントとして取得している.  
			

- create_stuff_documents_chain  
    この関数は、入力ドキュメントをリストとして受け取りフォーマットする.  
    [LINK](https://python.langchain.com/v0.1/docs/use_cases/chatbots/retrieval/#document-chains)

- create_retrieval_chain
    ベクトルストアから情報を取得し、全てのドキュメントを受け取り、単にすべてを一緒に詰める.  
    これによってドキュメントに基づくQ&Aが可能なのである.  

### 実行結果
```bash
{
	'input': 'what is Pinecone in machine learning?',
	 'context': [
		Document(id='18ae6a94-6dc3-44c7-9b3a-277146ec816d',
		 metadata={
			'source': 'mediumblog1.txt'
		},
		 page_content='Pinecone is designed to be fast and scalable,
		 allowing for efficient retrieval of similar data points based on their vector representations.\nIt can handle large-scale ML applications with millions or billions of data points.\nPinecone provides infrastructure management or maintenance to its users.\nPinecone can handle high query throughput and low latency search.\nPinecone is a secure platform that meets the security needs of businesses and organizations.\nPinecone is designed to be user-friendly and accessible via its simple API for storing and retrieving vector data,
		 making it easy to integrate into existing ML workflows.\nPinecone supports real-time updates,
		 allowing for efficient updates to the vector database as new data points are added. This ensures that the vector database remains up-to-date and accurate over time.\nPinecone can be synced with data from various sources using tools like Airbyte and monitored using Datadog\nChroma DB\nChroma DB is an open-source vector store for storing and retrieving vector embeddings. It is mainly used to save embeddings along with metadata to be used later by LLMs and can also be used for semantic search engines over text data.'),
		 Document(id='c088f890-30f1-4aa9-b3d9-f8d25dd2235b',
		 metadata={
			'source': 'mediumblog1.txt'
		},
		 page_content='Weaviate can store and search vectors from various data modalities,
		 including images,
		 text,
		 and audio.\nWeaviate provides seamless integration with machine learning frameworks such as Hugging Face,
		 Open AI,
		 LangChain,
		 Llamaindex,
		 TensorFlow,
		 PyTorch,
		 and Scikit-learn.\nWeaviate can index vectors in real-time,
		 making it ideal for applications that require low-latency search.\nWeaviate can be scaled to handle large volumes of data and high query throughput.\nWeaviate can be used in memory for fast search or with disk-based storage for larger datasets.\nWeaviate provides a user-friendly interface for managing vectors and performing searches.\nPinecone\nPinecone is a fully managed cloud-based vector database that is designed to make it easy for businesses and organizations to build and deploy large-scale ML applications.\n\n\nSome Pinecone Features : '),
		 Document(id='cd7e1c01-4252-4418-bfe2-4dc6d86afc18',
		 metadata={
			'source': 'mediumblog1.txt'
		},
		 page_content='ChatGPT prompts\n47 stories\n·\n1439 saves\nSparse embedding or BM25?\nInfiniFlow\nInfiniFlow\n\nSparse embedding or BM25?\nSince the open-sourcing of Infinity,
		 it has received a wide positive response from the community. Regarding the essential RAG technology we…\n7 min read\n·\nFeb 13,
		 2024\n16\n\nExperimenting with Vector Databases: Chromadb,
		 Pinecone,
		 Weaviate and Pgvector\nVishnu Sivan\nVishnu Sivan\n\nin\n\nCoinsBench\n\nExperimenting with Vector Databases: Chromadb,
		 Pinecone,
		 Weaviate and Pgvector\nVector databases are specialized systems designed for storing,
		 managing,
		 and searching embedding vectors. The widespread adoption of…\n10 min read\n·\nNov 14,
		 2023\n164\n\n1\n\n🚀 Blazing Fast Text Embeddings Inference for your RAG\nDavid Min\nDavid Min\n\n🚀 Blazing Fast Text Embeddings Inference for your RAG\nHugging Face\u200a—\u200aText Embeddings Inference (TEI)\n6 min read\n·\nOct 21,
		 2023\n94\n\n2\n\nVector Database\u200a—\u200aIntroduction and Python Implementation\nDenaya\nDenaya'),
		 Document(id='b80cf6b5-66cb-45f0-9bc4-2f09e297669f',
		 metadata={
			'source': 'mediumblog1.txt'
		},
		 page_content='Weaviate\nPinecone\nChroma DB\nQdrant\nMilvus\nHere’s an overview of some of the features of these vector databases. You can go see this comprehensive vector database features matrix by Dhruv Anand\n\n\nSource: Author\nWeaviate\nWeaviate is an open-source vector database that can be used to store,
		 search,
		 and manage vectors of any dimensionality. It is designed to be scalable and easy to use,
		 and it can be deployed on-premises or in the cloud.\n\n\nFeatures : ')
	],
	 'answer': 'Pinecone is a fully managed cloud-based vector database designed to facilitate the building and deployment of large-scale machine learning (ML) applications. It is optimized for fast and scalable retrieval of similar data points based on their vector representations. Pinecone can handle millions or billions of data points,
	 accommodates high query throughput,
	 and ensures low-latency search. The platform provides infrastructure management and maintenance,
	 real-time updates,
	 and easy integration into existing ML workflows via a simple API. Additionally,
	 Pinecone supports syncing with various data sources and monitoring,
	 making it a secure and user-friendly option for businesses and organizations.'
}
```