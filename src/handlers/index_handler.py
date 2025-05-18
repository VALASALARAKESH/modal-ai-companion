# src/handlers/index_handler.py
from typing import List, Dict, Optional
import pickle

import pathlib
from src.models.schemas import AgentConfig, volume
from src.services.file_service import FileService


class IndexHandler:
    def __init__(self):

        self.embedding_api = None
        self.file_service = FileService('/data')

    def initialize_embedding_api(self, agent_config: AgentConfig):
        from together import Together
        import os

        if self.embedding_api is None:
            self.embedding_api = Together(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key=os.environ["DEEP_INFRA_API_KEY"]
            )

    def get_paths(self, agent_config: AgentConfig) -> tuple[pathlib.Path, pathlib.Path]:
        """Get the paths for index and chunks files."""
        index_path = self.file_service.get_path(
            agent_config.workspace_id,
            f"{agent_config.agent_id}_index.ann"
        )
        chunks_path = self.file_service.get_path(
            agent_config.workspace_id,
            f"{agent_config.agent_id}_chunks.pkl"
        )
        return index_path, chunks_path

    def create_and_save_index(self, backstory: str, agent_config: AgentConfig, update_config: bool = False) -> bool:
        """Create and save vector index for background text."""
        from annoy import AnnoyIndex
        self.initialize_embedding_api(agent_config)
        index_path, chunks_path = self.get_paths(agent_config)

        # Check if index exists and update not required
        if index_path.exists() and chunks_path.exists() and not update_config:
            print("Using existing index (update_config is False)")
            return True

        print("Creating new index for background")
        try:
            embedded_chunks = self._embed_long_text(backstory)
            if not embedded_chunks:
                print("No embedded chunks generated")
                return False

            # Create and save index
            vector_length = len(embedded_chunks[0]["embedding"])
            index = AnnoyIndex(vector_length, 'angular')

            for i, chunk in enumerate(embedded_chunks):
                index.add_item(i, chunk["embedding"])

            index.build(3)  # Number of trees
            index.save(str(index_path))
            print(f"Saving index to {chunks_path}")
            # Save chunks separately
            with chunks_path.open("wb") as f:
                pickle.dump([chunk["chunk"] for chunk in embedded_chunks], f)
            print("Saved successfully")
            return True

        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False

    def search(self, query: str, agent_config: AgentConfig, n: int = 2) -> List[str]:
        from annoy import AnnoyIndex
        """Search for similar chunks in the index."""
        self.initialize_embedding_api(agent_config)
        index_path, chunks_path = self.get_paths(agent_config)

        if not index_path.exists() or not chunks_path.exists():
            print("Index or chunks not found")
            return []

        try:
            # Load chunks
            with chunks_path.open("rb") as f:
                chunks = pickle.load(f)

            # Generate query embedding
            query_embedding = self._get_embedding(query)

            # Load and search index
            index = AnnoyIndex(len(query_embedding), 'angular')
            index.load(str(index_path))

            similar_ids, distances = index.get_nns_by_vector(
                query_embedding,
                n,
                search_k=-1,  # Use default search_k
                include_distances=True
            )

            return [chunks[i] for i in similar_ids]

        except Exception as e:
            print(f"Error searching index: {str(e)}")
            return []

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        text = text.replace("\n", " ")
        response = self.embedding_api.embeddings.create(
            input=[text],
            model="sentence-transformers/all-MiniLM-L6-v2",
            encoding_format="float"
        )
        return response.data[0].embedding

    def _chunk_text(self, text: str, chunk_size: int = 50) -> List[str]:
        """Split text into chunks."""
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def _embed_long_text(self, text: str) -> List[Dict]:
        """Chunk and embed long text."""
        chunks = self._chunk_text(text)

        # Get embeddings for all chunks
        embeddings = []
        for chunk_batch in [chunks[i:i + 100] for i in range(0, len(chunks), 100)]:
            chunk_batch = [text.replace("\n", " ") for text in chunk_batch]
            response = self.embedding_api.embeddings.create(
                input=chunk_batch,
                model="sentence-transformers/all-MiniLM-L6-v2",
                encoding_format="float"
            )
            embeddings.extend([item.embedding for item in response.data])

        # Combine chunks with their embeddings
        return [
            {"chunk": chunk, "embedding": embedding}
            for chunk, embedding in zip(chunks, embeddings)
        ]