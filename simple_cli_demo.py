from rag_core import ingest_directory, answer

if __name__ == "__main__":
    print("[1/2] Ingesting ./data ...")
    n = ingest_directory("data")
    print(f"   -> Ingested {n} chunks\n")

    q = input("Enter a question: ")
    result = answer(q, k=5)
    print("\n[Answer]\n", result["answer"], "\n")
    print("[Top passages]")
    for p in result["passages"]:
        print(f"- {p['source']} (score {p['score']:.3f})\n  {p['text'][:200]}...\n")
