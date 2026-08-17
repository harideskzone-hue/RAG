import asyncio
import sys
sys.path.insert(0, ".")

from app.platform.config.config import config
config.mode = "native"

from app.tools.vector.store import get_vector_store, NativeVectorStore

store1 = get_vector_store()
store2 = get_vector_store()

print("store1 id:", id(store1), "type:", type(store1).__name__)
print("store2 id:", id(store2), "type:", type(store2).__name__)
print("Same instance?", store1 is store2)

async def main():
    # Simulate seeding store1
    from app.tools.vector.encoder import get_vector_encoder
    enc = get_vector_encoder()
    emb = enc.encode("green outfit person")
    await store1.insert("test_collection", [
        ["test-id-1"],
        [emb],
        ["cam_01"],
        ["2024-01-01T12:00:00"],
        ["Person wearing green outfit"],
        [None]
    ])
    print("Seeded store1. Metadata:", len(store1.metadata), "items")
    
    # Search store2 (simulating Dispatcher using a different instance)
    results = await store2.search("test_collection", emb, 5)
    print("store2 search results:", len(results), "items")
    
    # Search store1 directly
    results2 = await store1.search("test_collection", emb, 5)
    print("store1 search results:", len(results2), "items")

if __name__ == "__main__":
    asyncio.run(main())
