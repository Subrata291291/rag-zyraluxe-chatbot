from src.live_store import fetch_live_store

from src.product_search import (
    build_product_embeddings,
    filtered_semantic_search
)

from src.query_understanding import (
    understand_query
)

from src.knowledge_search import (
    build_knowledge_embeddings,
    search_knowledge
)

from src.recommendation_engine import (
    detect_recommendation_intent,
    get_recommendation,
    recommendation_context
)

from src.query_router import (
    classify_query
)

from src.conversation import (
    ConversationState
)

from src.answer_generator import (
    generate_product_answer,
    generate_knowledge_answer,
    generate_followup_answer,
    generate_normal_answer,
    stream_product_answer,
    stream_knowledge_answer,
    stream_followup_answer,
    stream_normal_answer
)

from src.answer_validator import (
    validate_answer
)

from src.config import (
    TOP_K_PRODUCTS,
    TOP_K_KNOWLEDGE
)


# ============================================================
# CHATBOT
# ============================================================

class JewelleryChatbot:

    def __init__(self, products=None, knowledge=None, conversation=None):

        print()
        print(
            "Loading jewellery chatbot..."
        )

        # ----------------------------------------------------
        # LOAD LIVE ZYRALUXE DATA
        # ----------------------------------------------------
        if products is not None and knowledge is not None:
            self.products = products
            self.knowledge = knowledge
            print(f"Reusing pre-loaded {len(self.products)} products and {len(self.knowledge)} knowledge documents.")
        else:
            self.products, self.knowledge = fetch_live_store()
            print(
                f"Loaded {len(self.products)} products."
            )
            print(
                f"Loaded {len(self.knowledge)} knowledge documents."
            )

        # ----------------------------------------------------
        # PRODUCT EMBEDDINGS
        # ----------------------------------------------------
        self.product_embeddings = (
            build_product_embeddings(
                self.products
            )
        )

        # ----------------------------------------------------
        # KNOWLEDGE EMBEDDINGS
        # ----------------------------------------------------
        if self.knowledge:
            self.knowledge_embeddings = (
                build_knowledge_embeddings(
                    self.knowledge
                )
            )
        else:
            self.knowledge_embeddings = []

        # ----------------------------------------------------
        # CONVERSATION
        # ----------------------------------------------------
        self.conversation = (
            conversation if conversation is not None else ConversationState()
        )

        print(
            "Chatbot ready."
        )


    # ========================================================
    # PRODUCT CONTEXT
    # ========================================================

    def product_context(
        self,
        results
    ):

        lines = []

        for rank, result in enumerate(
            results,
            start=1
        ):

            product = result["product"]

            lines.append(
                f"""
Product {rank}:
ID: {product['id']}
Name: {product['name']}
Category: {product['category']}
Metal: {product['metal']}
Karat: {product['karat']}
Price: ₹{product['price']}
Stock: {product.get('stock_status', 'not specified')}
SKU: {product.get('sku', '')}
Product URL: {product.get('url', '')}
Image URL: {product.get('image_url', '')}
Description: {product['description']}
Semantic Score: {result['score']:.3f}
"""
            )

        return "\n".join(lines)


    # ========================================================
    # KNOWLEDGE CONTEXT
    # ========================================================

    def knowledge_context(
        self,
        results
    ):

        lines = []

        for rank, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]

            lines.append(
                f"""
Document {rank}:
Document: {document['document']}
URL: {document.get('url', '')}
Score: {result['score']:.3f}

Content:
{document['content']}
"""
            )

        return "\n".join(lines)



    # ========================================================
    # RECOMMENDATION CONTEXT
    # ========================================================

    def add_recommendation_context(
        self,
        query,
        results,
        context
    ):
        """
        Detect and apply a deterministic recommendation on top
        of the already filtered/retrieved product results.

        The recommendation engine decides the product.
        The LLM only explains the supplied result.
        """

        intent = detect_recommendation_intent(query)

        if not intent or not results:
            return context

        print()
        print("===== RECOMMENDATION =====")
        print(f"Recommendation type: {intent}")

        recommendation = get_recommendation(
            results,
            intent
        )

        if not recommendation:
            print("No recommendation found.")
            return context

        product = recommendation["product"]

        print(
            f"Recommended: {product['name']} | "
            f"Rs. {product['price']:,}"
        )

        try:
            recommendation_info = recommendation_context(
                results,
                intent
            )
        except Exception as e:
            print(
                f"RECOMMENDATION CONTEXT ERROR: {e}"
            )
            return context

        if recommendation_info:
            context += "\\n\\n" + recommendation_info

        return context


    # ========================================================
    # STREAM ANSWER
    # ========================================================

    def ask_stream(self, query):
        """
        Stream the final LLM answer while preserving RAG safety.

        Search, filtering and query understanding happen first.
        The complete streamed answer is accumulated so it can be
        validated and stored in conversation memory.

        Yields text chunks to the caller. The caller can print each
        chunk immediately or send it to a web streaming response.
        """

        query = query.strip()

        if not query:
            yield "Please enter a question."
            return

        query_type = classify_query(
            query,
            has_previous_products=bool(
                self.conversation.last_products
            )
        )

        print()
        print(f"Query type: {query_type}")

        # --------------------------------------------------------
        # NORMAL CONVERSATION
        # --------------------------------------------------------
        if query_type == "normal":
            history = self.conversation.history_text()
            chunks = stream_normal_answer(query, history)
            parts = []

            try:
                for chunk in chunks:
                    parts.append(chunk)
                    yield chunk
            except Exception as e:
                print(f"\nNORMAL STREAM ERROR: {e}")
                if not parts:
                    yield "I'm sorry, I'm having trouble responding right now. Please try again."
                return

            answer = "".join(parts).strip()
            if answer:
                self.conversation.add_message(query, answer)
            return

        # --------------------------------------------------------
        # FOLLOW-UP
        # --------------------------------------------------------
        if query_type == "followup":
            context = self.conversation.product_context()
            history = self.conversation.history_text()

            if not context:
                yield "I don't have a previous product selection to refer to. Please tell me which jewellery you're interested in."
                return

            context = self.add_recommendation_context(
                query,
                self.conversation.last_products,
                context
            )

            chunks = stream_followup_answer(
                query,
                context,
                history
            )
            validation_context = context

        # --------------------------------------------------------
        # PRODUCT
        # --------------------------------------------------------
        elif query_type == "product":
            try:
                filters = understand_query(query)
                results = filtered_semantic_search(
                    query,
                    self.products,
                    self.product_embeddings,
                    filters,
                    TOP_K_PRODUCTS
                )
            except Exception as e:
                print(f"\nPRODUCT SEARCH ERROR: {e}")
                yield "I'm sorry, I couldn't search the product catalog right now."
                return

            if not results:
                yield "I couldn't find any products matching your requirements in the current catalog."
                return

            for rank, result in enumerate(results, start=1):
                product = result["product"]
                print(
                    f"{rank}. {product['name']} | "
                    f"Rs. {product['price']:,} | "
                    f"Score={result['score']:.3f}"
                )

            context = self.product_context(results)

            context = self.add_recommendation_context(
                query,
                results,
                context
            )

            validation_context = context
            self.conversation.set_products(results)
            chunks = stream_product_answer(
                query,
                context
            )

        # --------------------------------------------------------
        # KNOWLEDGE
        # --------------------------------------------------------
        elif query_type == "knowledge":
            if not self.knowledge:
                yield "The store knowledge documents are currently unavailable."
                return

            try:
                results = search_knowledge(
                    query,
                    self.knowledge,
                    self.knowledge_embeddings,
                    TOP_K_KNOWLEDGE
                )
            except Exception as e:
                print(f"\nKNOWLEDGE SEARCH ERROR: {e}")
                yield "I'm sorry, I couldn't search the store information right now."
                return

            if not results:
                yield "I couldn't find relevant information in the store knowledge documents."
                return

            for rank, result in enumerate(results, start=1):
                document = result["document"]
                print(
                    f"{rank}. {document['document']} | "
                    f"Score={result['score']:.3f}"
                )

            context = self.knowledge_context(results)
            validation_context = context
            chunks = stream_knowledge_answer(query, context)

        else:
            yield "I'm not sure how to handle that request. Please ask about our jewellery, products, or store policies."
            return

        # --------------------------------------------------------
        # STREAM + ACCUMULATE
        # --------------------------------------------------------
        parts = []

        try:
            for chunk in chunks:
                if chunk:
                    parts.append(chunk)
                    yield chunk
        except Exception as e:
            print(f"\nLLM STREAM ERROR: {e}")

            # Do not expose a second answer after partial output.
            # The caller has already received part of the response.
            return

        answer = "".join(parts).strip()

        if not answer:
            print("\nSTREAM ERROR: LLM returned empty answer.")
            return

        # --------------------------------------------------------
        # MEMORY
        # --------------------------------------------------------
        # In streaming mode, chunks have already been emitted in real time.
        # Save the answer directly to conversation memory without blocking
        # the stream completion with a redundant post-stream LLM call.
        self.conversation.add_message(
            query,
            answer
        )


    # ========================================================
    # HANDLE QUERY
    # ========================================================

    def ask(
        self,
        query
    ):

        query = query.strip()


        # ----------------------------------------------------
        # EMPTY QUERY
        # ----------------------------------------------------

        if not query:

            return (
                "Please enter a question."
            )


        # ----------------------------------------------------
        # CLASSIFY QUERY
        # ----------------------------------------------------

        query_type = classify_query(

            query,

            has_previous_products=bool(
                self.conversation.last_products
            )

        )


        print()

        print(
            f"Query type: {query_type}"
        )


        # ====================================================
        # NORMAL CONVERSATION
        # ====================================================

        if query_type == "normal":

            print()

            print(
                "===== NORMAL CONVERSATION ====="
            )


            history = (
                self.conversation.history_text()
            )


            try:

                answer = generate_normal_answer(

                    query=query,

                    history=history

                )


            except Exception as e:

                print()

                print(
                    "NORMAL CHAT ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I'm having trouble "
                    "responding right now. Please try again."
                )


            # ------------------------------------------------
            # NORMAL CHAT DOES NOT NEED RAG VALIDATION
            # ------------------------------------------------

            print()

            print(
                "Normal conversation - "
                "validation skipped."
            )


            self.conversation.add_message(

                query,

                answer

            )


            return answer


        # ====================================================
        # FOLLOW-UP
        # ====================================================

        if query_type == "followup":

            print()

            print(
                "===== FOLLOW-UP ====="
            )


            context = (
                self.conversation.product_context()
            )


            history = (
                self.conversation.history_text()
            )

            context = self.add_recommendation_context(
                query,
                self.conversation.last_products,
                context
            )


            # ------------------------------------------------
            # NO PREVIOUS PRODUCTS
            # ------------------------------------------------

            if not context:

                return (
                    "I don't have a previous product "
                    "selection to refer to. Please tell "
                    "me which jewellery you're interested in."
                )


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            try:

                answer = generate_followup_answer(

                    query,

                    context,

                    history

                )


            except Exception as e:

                print()

                print(
                    "FOLLOW-UP LLM ERROR:"
                )

                print(
                    str(e)
                )


                # IMPORTANT:
                # Do NOT continue to validation.

                return (
                    "I'm sorry, I couldn't process that "
                    "follow-up question right now. "
                    "Please try again."
                )


            # ------------------------------------------------
            # PROTECT AGAINST EMPTY ANSWER
            # ------------------------------------------------

            if not answer:

                print()

                print(
                    "FOLLOW-UP ERROR: "
                    "LLM returned an empty answer."
                )


                return (
                    "I'm sorry, I couldn't generate an "
                    "answer right now. Please try again."
                )


            validation_context = context


        # ====================================================
        # PRODUCT
        # ====================================================

        elif query_type == "product":

            print()

            print(
                "===== PRODUCT SEARCH ====="
            )


            try:

                # ------------------------------------------------
                # UNDERSTAND USER QUERY
                # ------------------------------------------------

                query_info = understand_query(query)

                print()
                print("===== QUERY UNDERSTANDING =====")

                print(
                    f"Original query: {query}"
                )

                print(
                    f"Corrected query: "
                    f"{query_info.get('corrected_query', query)}"
                )

                print(
                    f"Category: "
                    f"{query_info.get('category')}"
                )

                print(
                    f"Metal: "
                    f"{query_info.get('metal')}"
                )

                print(
                    f"Material type: "
                    f"{query_info.get('material_type')}"
                )

                print(
                    f"Karat: "
                    f"{query_info.get('karat')}"
                )

                print(
                    f"Min price: "
                    f"{query_info.get('min_price')}"
                )

                print(
                    f"Max price: "
                    f"{query_info.get('max_price')}"
                )

                # ------------------------------------------------
                # USE CORRECTED QUERY FOR SEMANTIC SEARCH
                # ------------------------------------------------

                search_query = query_info.get(
                    "corrected_query",
                    query
                )

                if not search_query:
                    search_query = query

                # ------------------------------------------------
                # FILTER + SEMANTIC SEARCH
                # ------------------------------------------------

                results = filtered_semantic_search(
                    query=search_query,
                    products=self.products,
                    product_embeddings=self.product_embeddings,
                    filters=query_info,
                    top_k=TOP_K_PRODUCTS
                )

            except Exception as e:

                print()

                print(
                    "PRODUCT SEARCH ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't search the "
                    "product catalog right now."
                )


            # ------------------------------------------------
            # NO RESULTS
            # ------------------------------------------------

            if not results:

                return (
                    "I couldn't find any products "
                    "matching your requirements."
                )


            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            for rank, result in enumerate(

                results,

                start=1

            ):

                product = result["product"]


                print(
                    f"{rank}. "
                    f"{product['name']} | "
                    f"Rs. {product['price']:,} | "
                    f"Score={result['score']:.3f}"
                )


            # ------------------------------------------------
            # CONTEXT
            # ------------------------------------------------

            context = (
                self.product_context(
                    results
                )
            )

            context = self.add_recommendation_context(
                query,
                results,
                context
            )


            # ------------------------------------------------
            # LLM
            # ------------------------------------------------

            try:

                answer = generate_product_answer(

                    query,

                    context

                )


            except Exception as e:

                print()

                print(
                    "PRODUCT LLM ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't generate "
                    "a product answer right now. "
                    "Please try again."
                )


            # ------------------------------------------------
            # EMPTY ANSWER
            # ------------------------------------------------

            if not answer:

                return (
                    "I'm sorry, I couldn't generate "
                    "an answer right now. Please try again."
                )


            validation_context = context


            # ------------------------------------------------
            # SAVE PRODUCTS
            # ------------------------------------------------

            self.conversation.set_products(
                results
            )


        # ====================================================
        # KNOWLEDGE
        # ====================================================

        elif query_type == "knowledge":

            print()

            print(
                "===== KNOWLEDGE SEARCH ====="
            )


            if not self.knowledge:

                return (
                    "The store knowledge documents "
                    "are currently unavailable."
                )


            # ------------------------------------------------
            # SEARCH
            # ------------------------------------------------

            try:

                results = search_knowledge(

                    query,

                    self.knowledge,

                    self.knowledge_embeddings,

                    TOP_K_KNOWLEDGE

                )


            except Exception as e:

                print()

                print(
                    "KNOWLEDGE SEARCH ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't search the "
                    "store information right now."
                )


            # ------------------------------------------------
            # NO RESULTS
            # ------------------------------------------------

            if not results:

                return (
                    "I couldn't find relevant information "
                    "in the store knowledge documents."
                )


            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            for rank, result in enumerate(

                results,

                start=1

            ):

                document = result["document"]


                print(

                    f"{rank}. "

                    f"{document['document']} | "

                    f"Score={result['score']:.3f}"

                )


            # ------------------------------------------------
            # CONTEXT
            # ------------------------------------------------

            context = (
                self.knowledge_context(
                    results
                )
            )


            # ------------------------------------------------
            # LLM
            # ------------------------------------------------

            try:

                answer = generate_knowledge_answer(

                    query,

                    context

                )


            except Exception as e:

                print()

                print(
                    "KNOWLEDGE LLM ERROR:"
                )

                print(
                    str(e)
                )


                return (
                    "I'm sorry, I couldn't generate "
                    "an answer from the store information "
                    "right now. Please try again."
                )


            # ------------------------------------------------
            # EMPTY ANSWER
            # ------------------------------------------------

            if not answer:

                return (
                    "I'm sorry, I couldn't generate "
                    "an answer right now. Please try again."
                )


            validation_context = context


        # ====================================================
        # UNKNOWN TYPE
        # ====================================================

        else:

            return (
                "I'm not sure how to handle that request. "
                "Please ask about our jewellery, products, "
                "or store policies."
            )


        # ====================================================
        # ANSWER VALIDATION
        # ====================================================

        print()

        print(
            "===== ANSWER VALIDATION ====="
        )


        # ----------------------------------------------------
        # FINAL SAFETY CHECK
        # ----------------------------------------------------

        if not answer:

            print(
                "VALIDATION SKIPPED: "
                "No answer was generated."
            )


            return (
                "I couldn't generate an answer right now. "
                "Please try again."
            )


        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        try:

            validation_result = (
                validate_answer(

                    query,

                    validation_context,

                    answer

                )
            )


        except Exception as e:

            print()

            print(
                "Validation error:"
            )

            print(
                str(e)
            )


            return (
                "I generated an answer, but I couldn't "
                "verify it against the available store "
                "information. Please try again."
            )


        # ----------------------------------------------------
        # VALIDATION RESULT SAFETY CHECK
        # ----------------------------------------------------

        if (
            validation_result is None
            or not isinstance(
                validation_result,
                tuple
            )
            or len(validation_result) != 2
        ):

            print()

            print(
                "VALIDATION ERROR: "
                "Invalid validator response."
            )


            return (
                "I couldn't verify the answer against "
                "the available store information."
            )


        valid, validation_message = (
            validation_result
        )


        # ====================================================
        # VALIDATION PASSED
        # ====================================================

        if valid:

            print(
                "VALIDATION: PASSED"
            )

            final_answer = answer


        # ====================================================
        # VALIDATION FAILED
        # ====================================================

        else:

            print(
                "VALIDATION: FAILED"
            )

            print(
                validation_message
            )


            final_answer = (
                "I couldn't verify that answer "
                "from the available store information."
            )


        # ====================================================
        # MEMORY
        # ====================================================

        self.conversation.add_message(

            query,

            final_answer

        )


        # ====================================================
        # RETURN
        # ====================================================

        return final_answer