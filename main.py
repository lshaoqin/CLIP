from fastapi import FastAPI, File, UploadFile
from milvus_utils import connect_to_milvus, create_milvus_collection, upsert_milvus, generate_entities, create_milvus_index, query_milvus, check_collection_exists, drop_milvus_collection
from typing import List
import helpers
import base64
import asyncio

# To start, run uvicorn main:app --reload
app = FastAPI()
connect_to_milvus()

model, preprocess, device = helpers.instantiate_model()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/{collection}/create")
async def create_collection(collection: str, embeddings_len: int = 512, labels_len: int = 5000):
    try:
        create_milvus_collection(collection, embeddings_len, labels_len)
    except Exception as e:
        return {"error": str(e)}
    return {"result": "Collection created."}

@app.get("/{collection}/drop")
async def drop_collection(collection: str):
    try:
        result = drop_milvus_collection(collection)
    except Exception as e:
        return {"error": str(e)}
    return {"result": str(result)}

@app.post("/{collection}/upsert_text")
async def upsert_txt(collection: str, inputs: List[str]):
    try:
        if not check_collection_exists(collection):
            create_milvus_collection(collection, 512, 5000)
        
        embeddings = helpers.generate_text_embeddings(inputs, model, device)

        entities = generate_entities(embeddings, inputs)
        result = upsert_milvus(entities, collection)

    except Exception as e:
        return {"error": str(e)}

    return {"result": str(result)}

@app.post("/{collection}/upsert_images")
async def upsert_images(collection: str, files: List[UploadFile]):
    try:
        if not check_collection_exists(collection):
            create_milvus_collection(collection, 512, 5000)
        
         # Save uploaded files to disk
        file_paths = []
        for file in files:
            file_path = f"./.assets/test_images/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            file_paths.append(file_path)

        embeddings = helpers.generate_image_embeddings(file_paths, model, preprocess, device)

        entities = generate_entities(embeddings, [file.filename for file in files])
        result = upsert_milvus(entities, collection)

    except Exception as e:
        return {"error": str(e)}

    return {"result": str(result)}

@app.get("/{collection}/create_index")
async def create_index(collection: str):
    try:
        if not check_collection_exists(collection):
            return {"error": "Collection does not exist."}

        create_milvus_index(collection, "embeddings", "IVF_FLAT", "L2", {"nlist": 128})

    except Exception as e:
        return {"error": str(e)}

    return {"result": "Index created."}

@app.post("/{collection}/query")
async def query(collection: str, query: str):
    try:
        if not check_collection_exists(collection):
            return {"error": "Collection does not exist."}

        embeddings = helpers.generate_text_embeddings([query], model, device)
        embeddings = embeddings.numpy()
        if len(embeddings[0].shape) == 1:
            embeddings = embeddings[0].reshape(1, -1)
        results = query_milvus(collection, embeddings, "embeddings", {"metric_type": "L2", "params": {"nprobe": 10}})

        # for result in results:
        #     result["labels"] = result["labels"].decode("utf-8")
        #     if result["labels"].endswith(".jpg"):
        #         with open(f"./.assets/test_images/{result['labels']}", "rb") as image_file:
        #             encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        #         result["labels"] = encoded_string
        #         result["isImage"] = True
        #     else:
        #         result["isImage"] = False

    except Exception as e:
        return {"error": str(e)}

    return {"result": str(results)}

async def main():
    await drop_collection("demo")
    await create_collection("demo")

if __name__ == '__main__':
    asyncio.run(main())