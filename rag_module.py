from pinecone import Pinecone

class RAGQueryModule:
    def __init__(self, api_key, index_name, model="multilingual-e5-large"):
        # Initialize Pinecone and connect to the existing index
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.model = model
        self.index = self.pc.Index(index_name)

    def query_context(self, key_entities, request_transcription, top_k=3):
        # Combine key entities and transcription into a query
        query = f"{key_entities}. {request_transcription}"
        
        # Generate embedding for the combined query
        query_embedding = self.pc.inference.embed(
            model=self.model,
            inputs=[query],
            parameters={"input_type": "query"}
        )[0]['values']

        # Query the index for similar chunks
        results = self.index.query(
            namespace="dispatcher-namespace",
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

        # Display results
        print("Query Results:")
        for match in results["matches"]:
            print(f"Score: {match['score']}")
            print(f"Text: {match['metadata']['text']}\n")
        
        return results

# Example usage:
api_key = "7623f706-02e2-427e-8e10-c1b77db64b56"
index_name = "dispatcher-index"
rag_query_module = RAGQueryModule(api_key=api_key, index_name=index_name)

# Sample query with key entities and transcription
key_entities = "headache, dizziness"
request_transcription = "I feel lightheaded and my head hurts."
results = rag_query_module.query_context(key_entities, request_transcription)
