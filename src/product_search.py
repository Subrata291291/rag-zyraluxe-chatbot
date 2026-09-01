import re


# ============================================================
# PRODUCT TEXT
# ============================================================

def product_to_text(product):
    """
    Convert a product into searchable text.

    We include all important product attributes so that
    semantic search can understand the product.
    """

    parts = [
        f"Product ID: {product.get('id', '')}",
        f"Name: {product.get('name', '')}",
        f"Category: {product.get('category', '')}",
        f"Metal: {product.get('metal', '')}",
        f"Karat: {product.get('karat', '')}",
        f"Price: ₹{product.get('price', '')}",
        f"Description: {product.get('description', '')}"
    ]

    # Optional field for future gold-plated / silver-plated etc.
    if product.get("material_type"):

        parts.append(
            f"Material Type: {product['material_type']}"
        )

    return ". ".join(parts)


# ============================================================
# BUILD PRODUCT EMBEDDINGS
# ============================================================

def build_product_embeddings(products):
    """
    Build embeddings for all products.
    """

    # Kept for backward compatibility. Retrieval is intentionally
    # lightweight so the API can run on small Render instances.
    return []


# ============================================================
# NORMALIZE VALUE
# ============================================================

def normalize_value(value):
    """
    Normalize strings so that comparisons are reliable.

    Example:

        'Gold'       -> 'gold'
        ' GOLD '     -> 'gold'
        '22k'        -> '22k'
        '22K'        -> '22k'
    """

    if value is None:
        return None

    return str(value).strip().lower()


# ============================================================
# METADATA FILTER
# ============================================================

def filter_products(
    products,
    filters
):
    """
    Apply exact metadata filters.

    Expected filters:

        {
            "category": "ring",
            "metal": "gold",
            "karat": "22K",
            "min_price": None,
            "max_price": 20000
        }

    IMPORTANT:

    Metadata filtering happens BEFORE semantic search.

    This prevents semantic search from returning products
    that violate hard customer requirements.
    """

    if not filters:

        return products

    filtered = []

    category = normalize_value(
        filters.get("category")
    )

    metal = normalize_value(
        filters.get("metal")
    )

    karat = normalize_value(
        filters.get("karat")
    )

    min_price = filters.get(
        "min_price"
    )

    max_price = filters.get(
        "max_price"
    )

    material_type = normalize_value(
        filters.get("material_type")
    )


    # ========================================================
    # CHECK EACH PRODUCT
    # ========================================================

    for product in products:

        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        if category:

            product_category = normalize_value(
                product.get("category")
            )

            if product_category != category:

                continue


        # ----------------------------------------------------
        # METAL
        # ----------------------------------------------------

        if metal:

            product_metal = normalize_value(
                product.get("metal")
            )

            if product_metal != metal:

                continue


        # ----------------------------------------------------
        # KARAT
        # ----------------------------------------------------

        if karat:

            product_karat = normalize_value(
                product.get("karat")
            )

            if product_karat != karat:

                continue


        # ----------------------------------------------------
        # MATERIAL TYPE
        # ----------------------------------------------------

        if material_type:

            product_material_type = normalize_value(
                product.get("material_type")
            )

            if product_material_type != material_type:

                continue


        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        try:

            product_price = float(
                product.get("price", 0)
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        # ----------------------------------------------------
        # MIN PRICE
        # ----------------------------------------------------

        if min_price is not None:

            if product_price < float(min_price):

                continue


        # ----------------------------------------------------
        # MAX PRICE
        # ----------------------------------------------------

        if max_price is not None:

            if product_price > float(max_price):

                continue


        # ----------------------------------------------------
        # PRODUCT PASSED ALL FILTERS
        # ----------------------------------------------------

        filtered.append(product)


    return filtered


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def _tokens(value):
    return set(
        re.findall(
            r"[a-z0-9]+",
            str(value or "").lower(),
        )
    )


def _rank_products(query, products, top_k):
    """Rank catalogue products without loading an embedding model."""

    query_tokens = _tokens(query)
    ranked = []

    for product in products:
        name_tokens = _tokens(product.get("name"))
        category_tokens = _tokens(product.get("category"))
        details_tokens = _tokens(
            " ".join(
                str(product.get(key, ""))
                for key in ("metal", "karat", "description", "sku")
            )
        )
        score = float(
            5 * len(query_tokens & name_tokens)
            + 3 * len(query_tokens & category_tokens)
            + len(query_tokens & details_tokens)
        )
        ranked.append((score, product))

    ranked.sort(
        key=lambda item: (
            item[0],
            bool(item[1].get("is_in_stock", False)),
            -float(item[1].get("price", 0) or 0),
        ),
        reverse=True,
    )

    return [
        {"product": product, "score": score}
        for score, product in ranked[:max(1, int(top_k))]
    ]

def semantic_search(
    query,
    products,
    product_embeddings,
    top_k=5
):
    """
    Perform semantic search across products.

    This function is kept for compatibility with the
    existing chatbot.

    For production usage where filters are available,
    prefer:

        filtered_semantic_search()
    """

    if not products:

        return []

    return _rank_products(query, products, top_k)


    # --------------------------------------------------------
    # CREATE QUERY EMBEDDING
    # --------------------------------------------------------

    query_embedding = embed_text(
        query
    )


    # --------------------------------------------------------
    # CALCULATE COSINE SIMILARITY
    # --------------------------------------------------------

    scores = cosine_similarity(
        [query_embedding],
        product_embeddings
    )[0]


    # --------------------------------------------------------
    # SORT BY SCORE
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        scores
    )[::-1]


    # --------------------------------------------------------
    # BUILD RESULTS
    # --------------------------------------------------------

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "product": products[index],
                "score": float(
                    scores[index]
                )
            }
        )


    return results


# ============================================================
# FILTERED + SEMANTIC SEARCH
# ============================================================

def filtered_semantic_search(
    query,
    products,
    product_embeddings,
    filters,
    top_k=5
):
    """
    Production-style product retrieval.

    Pipeline:

        1. Apply exact metadata filters
        2. Keep only matching products
        3. Run semantic search on those products
        4. Return top results

    This prevents semantic search from overriding
    hard customer requirements.
    """

    # ========================================================
    # STEP 1 — METADATA FILTER
    # ========================================================

    filtered_products = filter_products(
        products,
        filters
    )


    if not filtered_products:

        return []

    return _rank_products(
        query,
        filtered_products,
        top_k,
    )


    # ========================================================
    # STEP 2 — FIND ORIGINAL EMBEDDING INDICES
    # ========================================================

    filtered_indices = []

    for product in filtered_products:

        for index, original_product in enumerate(
            products
        ):

            if (
                original_product.get("id")
                == product.get("id")
            ):

                filtered_indices.append(
                    index
                )

                break


    # ========================================================
    # STEP 3 — GET FILTERED EMBEDDINGS
    # ========================================================

    filtered_embeddings = (
        product_embeddings[
            filtered_indices
        ]
    )


    # ========================================================
    # STEP 4 — SEMANTIC SEARCH
    # ========================================================

    query_embedding = embed_text(
        query
    )


    scores = cosine_similarity(
        [query_embedding],
        filtered_embeddings
    )[0]


    # ========================================================
    # STEP 5 — RANK
    # ========================================================

    ranked_indices = np.argsort(
        scores
    )[::-1]


    # ========================================================
    # STEP 6 — BUILD RESULTS
    # ========================================================

    results = []

    for local_index in ranked_indices[:top_k]:

        original_index = filtered_indices[
            local_index
        ]

        results.append(
            {
                "product": products[
                    original_index
                ],

                "score": float(
                    scores[local_index]
                )
            }
        )


    return results
