# Connect to milvus server
# Credit to this tutorial by Stephen Collins for information on setting up milvus and text embedding
# https://dev.to/stephenc222/how-to-use-milvus-to-store-and-query-vector-embeddings-5hhl
from pymilvus import connections, utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import time

def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise e


def create_milvus_collection(name, embeddings_len, labels_len):
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embeddings_len),
        FieldSchema(name="labels", dtype=DataType.VARCHAR, max_length=labels_len),
        FieldSchema(name="timestamp", dtype=DataType.INT64)
    ]
    description = name + "embeddings"
    collection = Collection(name, CollectionSchema(fields, description), consistency_level="Strong")
    print(f"Collection {name} created.")
    return collection

def drop_milvus_collection(name):
    if utility.has_collection(name):
        collection = Collection(name)
        collection.drop()
        return f"Collection {name} dropped."
    else:
        return f"Collection {name} does not exist."

def generate_entities(embeddings, labels, times = None):
    if times is None:
        times = [int(time.time()) for _ in range(len(embeddings))]
    entities = [
        embeddings,
        labels,
        times
    ]
    return entities

def upsert_milvus(entities, name):
    collection = Collection(name)
    insert_result = collection.insert(entities)
    print(f"Inserted {len(entities[0])} entities.")
    return insert_result

def create_milvus_index(collection_name, field_name, index_type, metric_type, params):
    collection = Collection(collection_name)
    index = {"index_type": index_type, "metric_type": metric_type, "params": params}
    collection.create_index(field_name, index)
    print(f"Index created for {collection_name}.")

def query_milvus(collection, search_vectors, search_field, search_params):
    if isinstance(collection, str):
        collection = Collection(collection)

    collection.load()
    result = collection.search(search_vectors, search_field, search_params, limit=3, output_fields=["labels"])
    return result[0]

def check_collection_exists(name):
    if utility.has_collection(name):
        print(f"Collection {name} exists.")
        return True
    else:
        print(f"Collection {name} does not exist.")
        return False

if __name__ == "__main__":
    connect_to_milvus()
    create_milvus_collection("ImageBind", 1024, 5000)