from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from starpoint.db import Client
from os import environ

import clip
import torch

api_key = environ.get("STARPOINT_API_KEY")
if api_key is None:
    raise ValueError("STARPOINT_API_KEY environment variable must be set")

collection_id = environ.get("STARPOINT_COLLECTION_ID")
if collection_id is None:
    raise ValueError("STARPOINT_COLLECTION_ID environment variable must be set")

client = Client(api_key=api_key)

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def embed_text(text: str) -> List[float]:
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(text).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        return text_encoded.tolist()[0]


@app.get("/api/v1/query", response_model=None)
def query(text: Optional[str] = None) -> Response | Dict[str, Any]:
    print("embedding", text)
    if text is not None:
        embedding = embed_text(text)
        results = client.query(
            collection_id=collection_id,
            query_embedding=embedding,
        )
        
        print(results.get("results"))
        return results
        # print(len(embedding))
        # return SimilarProductsResponse(text=text, embedding=embedding)


    # return a 400 error
    return Response(status_code=status.HTTP_400_BAD_REQUEST)
