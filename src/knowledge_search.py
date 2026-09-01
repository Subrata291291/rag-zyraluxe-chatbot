import re


# ============================================================
# BUILD KNOWLEDGE EMBEDDINGS
# ============================================================

def build_knowledge_embeddings(
    documents
):

    # Kept for compatibility with the chatbot interface.  The live
    # service deliberately uses lightweight lexical retrieval.
    return []


# ============================================================
# SEARCH KNOWLEDGE
# ============================================================

def search_knowledge(
    query,
    documents,
    embeddings,
    top_k=3
):

    if not documents:

        return []

    query_tokens = set(
        re.findall(r"[a-z0-9]+", str(query or "").lower())
    )

    ranked = []

    for document in documents:
        document_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                " ".join(
                    str(document.get(key, ""))
                    for key in ("document", "content", "url")
                ).lower(),
            )
        )
        ranked.append((float(len(query_tokens & document_tokens)), document))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return [
        {"document": document, "score": score}
        for score, document in ranked[:max(1, int(top_k))]
    ]

    query_embedding = embed_text(query)

    scores = cosine_similarity(
        [query_embedding],
        embeddings
    )[0]

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "document": documents[index],
                "score": float(scores[index])
            }
        )

    return results
