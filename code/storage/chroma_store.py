"""ChromaDB persistent store — three collections, one per company. Embeddings supplied externally."""

from __future__ import annotations

from typing import Optional

import chromadb


COLLECTIONS = ("hackerrank", "claude", "visa")


class ChromaStore:
    def __init__(self, persist_path: str = "chroma_store"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self._cols: dict = {}
        for c in COLLECTIONS:
            self._cols[c] = self.client.get_or_create_collection(
                name=c,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )

    def clear_company(self, company: str) -> None:
        try:
            self.client.delete_collection(name=company)
        except Exception:
            pass
        self._cols[company] = self.client.get_or_create_collection(
            name=company,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add_chunks(
        self,
        company: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        col = self._cols[company]
        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(
        self,
        company: str,
        query_embedding: list[float],
        top_k: int = 8,
        where: Optional[dict] = None,
    ) -> list[dict]:
        col = self._cols[company]
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where
        try:
            res = col.query(**kwargs)
        except Exception:
            res = col.query(query_embeddings=[query_embedding], n_results=top_k)
        out = []
        if not res.get("ids") or not res["ids"][0]:
            return []
        for i in range(len(res["ids"][0])):
            out.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i] if res.get("distances") else 1.0,
            })
        return out

    def count(self, company: str) -> int:
        return self._cols[company].count()
