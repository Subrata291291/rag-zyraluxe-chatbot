import os
import time
import json
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.chatbot import JewelleryChatbot
from src.live_store import fetch_live_store


# ============================================================
# CONFIG
# ============================================================

APP_ENV = os.getenv("APP_ENV", "development").lower()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ============================================================
# CORS ORIGINS
# ============================================================
#
# Production WordPress site:
#   https://zyraluxe.in
#   https://www.zyraluxe.in
#
# Local development:
#   http://localhost
#   http://127.0.0.1
#   http://localhost:3000
#   http://localhost:5173
#
# You can override this using the CORS_ORIGINS
# environment variable in production.
#

DEFAULT_CORS_ORIGINS = (
    "https://zyraluxe.in,"
    "https://www.zyraluxe.in,"
    "http://localhost,"
    "http://127.0.0.1,"
    "http://localhost:3000,"
    "http://localhost:5173"
)

CORS_ORIGINS = [
    "https://zyraluxe.in",
    "https://www.zyraluxe.in",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://localhost:5173",
]


# ============================================================
# OPTIONAL APPLICATION ACCESS KEY
# ============================================================

# Optional application-level key.
#
# IMPORTANT:
# Do not put a secret API key into browser JavaScript.
#
# For a public WordPress widget, leave this empty and use
# your web-server/proxy/rate-limit layer for protection.

API_ACCESS_KEY = os.getenv(
    "CHATBOT_API_ACCESS_KEY",
    ""
).strip()


# ============================================================
# RATE LIMITING
# ============================================================

RATE_LIMIT_REQUESTS = int(
    os.getenv(
        "RATE_LIMIT_REQUESTS",
        "30"
    )
)

RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv(
        "RATE_LIMIT_WINDOW_SECONDS",
        "60"
    )
)


# ============================================================
# MESSAGE LIMIT
# ============================================================

MAX_MESSAGE_LENGTH = int(
    os.getenv(
        "MAX_MESSAGE_LENGTH",
        "4000"
    )
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Zyraluxe Jewellery AI Assistant API",
    version="1.1.0",
    description="RAG-powered jewellery shopping assistant."
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    # Zyraluxe WordPress website
    allow_origins=[
        "https://zyraluxe.in",
        "https://www.zyraluxe.in",

        # Local testing
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://localhost:5173",
    ],

    allow_credentials=False,

    # Allow all HTTP methods
    allow_methods=["*"],

    # Allow all request headers
    allow_headers=["*"],

    # Expose useful response headers
    expose_headers=[
        "X-Request-ID",
    ],
)


# ============================================================
# SESSION STORE
# ============================================================

# Development-only storage.
#
# Each session currently owns a JewelleryChatbot because the
# chatbot contains ConversationState.
#
# For production with multiple workers, replace this with
# a shared session store such as Redis and avoid relying on
# process-local memory.

sessions: Dict[str, JewelleryChatbot] = {}


# ============================================================
# RATE LIMIT STORE
# ============================================================

# Development/local-process rate limiter.
# Keyed by client IP.

rate_history: Dict[
    str,
    Deque[float]
] = defaultdict(deque)


# ============================================================
# RATE LIMIT CHECK
# ============================================================

def check_rate_limit(
    client_ip: str
):

    now = time.monotonic()

    cutoff = (
        now -
        RATE_LIMIT_WINDOW_SECONDS
    )

    timestamps = rate_history[
        client_ip
    ]

    while (
        timestamps
        and timestamps[0] < cutoff
    ):

        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUESTS:

        raise HTTPException(
            status_code=429,
            detail=(
                "Too many requests. "
                "Please try again later."
            )
        )

    timestamps.append(now)


# ============================================================
# ACCESS CONTROL
# ============================================================

def check_access(
    request: Request
):

    """
    Optional application-level access key.

    Do not treat this as a browser secret. If the widget is
    calling the API directly from a public website, a secret
    key embedded in JavaScript can be extracted by users.
    """

    if not API_ACCESS_KEY:

        return

    supplied_key = request.headers.get(
        "X-Chatbot-Key",
        ""
    ).strip()

    if supplied_key != API_ACCESS_KEY:

        raise HTTPException(
            status_code=401,
            detail="Unauthorized."
        )


# ============================================================
# SESSION
# ============================================================

def get_or_create_session(
    session_id: str | None
) -> Tuple[str, JewelleryChatbot]:

    # Existing session

    if session_id:

        chatbot = sessions.get(
            session_id
        )

        if chatbot is not None:

            return (
                session_id,
                chatbot
            )

    # New session

    new_session_id = str(
        uuid4()
    )

    chatbot = JewelleryChatbot()

    sessions[
        new_session_id
    ] = chatbot

    return (
        new_session_id,
        chatbot
    )


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH
    )

    session_id: str | None = Field(
        default=None,
        max_length=100
    )


# ============================================================
# RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):

    session_id: str

    answer: str


# ============================================================
# REQUEST MIDDLEWARE
# ============================================================

@app.middleware("http")
async def request_guard(
    request: Request,
    call_next
):

    request_id = str(
        uuid4()
    )

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        # CORS preflight requests are OPTIONS requests.
        #
        # Do not apply chatbot access/rate limiting logic
        # to the OPTIONS request.
        # ----------------------------------------------------

        if request.method != "OPTIONS":

            check_access(
                request
            )

            # Rate-limit API endpoints only.

            if request.url.path in {
                "/chat",
                "/chat/stream"
            }:

                check_rate_limit(
                    client_ip
                )

        response = await call_next(
            request
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        return response

    except HTTPException:

        raise

    except Exception:

        # Do not leak internal exceptions
        # to clients.

        raise HTTPException(
            status_code=500,
            detail="Internal server error."
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def root():
    """Provide a useful response when the Render service URL is opened."""

    return {
        "service": "Zyraluxe Jewellery AI Assistant API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "zyraluxe-jewellery-ai",
        "environment": APP_ENV
    }


# ============================================================
# NORMAL CHAT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(
    request: ChatRequest
):

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        session_id, chatbot = (
            get_or_create_session(
                request.session_id
            )
        )

        answer = chatbot.ask(
            message
        )

        if not answer:

            raise RuntimeError(
                "Chatbot returned an empty answer."
            )

        return ChatResponse(
            session_id=session_id,
            answer=answer
        )

    except HTTPException:

        raise

    except Exception as exc:

        print(
            f"CHAT ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process the request."
        ) from exc


# ============================================================
# PRODUCT PAYLOAD FOR FRONTEND
# ============================================================

def build_product_payload(
    chatbot: JewelleryChatbot
):
    """
    Convert the chatbot's latest product selection into a small,
    JSON-safe payload for the WordPress frontend.

    The chatbot stores products as search-result dictionaries:
        {
            "product": {...},
            "score": ...
        }

    We intentionally expose only product/catalog fields that are
    useful for rendering product cards.
    """

    try:
        results = getattr(
            chatbot.conversation,
            "last_products",
            []
        )

        if not results:
            return []

        products = []

        for result in results:

            if not isinstance(result, dict):
                continue

            product = result.get(
                "product",
                result
            )

            if not isinstance(product, dict):
                continue

            item = {}

            # ----------------------------------------------------
            # Core catalog fields
            # ----------------------------------------------------

            fields = [
                "id",
                "name",
                "category",
                "metal",
                "karat",
                "price",
                "description",
                "sku",
                "slug",
                "url",
                "product_url",
                "permalink",
                "image",
                "image_url",
                "thumbnail",
                "thumbnail_url",
                "images",
                "stock_status",
                "is_in_stock",
                "stock_quantity",
                "on_sale",
                "rating",
                "review_count",
                "attributes",
                "tags",
            ]

            for field in fields:

                if field in product:
                    item[field] = product[field]

            # ----------------------------------------------------
            # Keep semantic score when available
            # ----------------------------------------------------

            if "score" in result:
                item["score"] = result["score"]

            # ----------------------------------------------------
            # Do not lose a product if the catalog uses an
            # unexpected structure. At minimum, preserve its
            # actual product dictionary.
            # ----------------------------------------------------

            if not item:
                item = product.copy()

            products.append(item)

        return products

    except Exception as exc:

        print(
            f"PRODUCT PAYLOAD ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        return []


def product_ids(
    products
):
    """
    Return stable product IDs for detecting whether a new
    product search changed the current selection.
    """

    ids = []

    for product in products:

        if not isinstance(product, dict):
            continue

        product_id = product.get("id")

        if product_id is not None:
            ids.append(str(product_id))
            continue

        # Fallback when the catalog does not expose an ID.
        fallback = (
            product.get("sku")
            or product.get("slug")
            or product.get("name")
        )

        if fallback is not None:
            ids.append(str(fallback))

    return ids


# ============================================================
# STREAMING CHAT
# ============================================================

@app.post(
    "/chat/stream"
)
def chat_stream(
    request: ChatRequest
):

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        session_id, chatbot = (
            get_or_create_session(
                request.session_id
            )
        )

    except Exception as exc:

        print(
            f"SESSION ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to initialize "
                "chat session."
            )
        ) from exc


    # ========================================================
    # SSE EVENT GENERATOR
    # ========================================================

    def event_generator():

        # ----------------------------------------------------
        # Capture the product selection BEFORE this request.
        # This prevents the frontend from receiving duplicate
        # product cards for ordinary follow-up messages.
        # ----------------------------------------------------

        before_products = build_product_payload(
            chatbot
        )

        before_ids = product_ids(
            before_products
        )

        # ----------------------------------------------------
        # Send session ID first
        # ----------------------------------------------------

        yield (
            "event: session\n"
            f"data: {session_id}\n\n"
        )

        try:

            # ------------------------------------------------
            # Stream chatbot response
            # ------------------------------------------------

            for chunk in chatbot.ask_stream(
                message
            ):

                if not chunk:
                    continue

                # ------------------------------------------------
                # SSE requires every line of a multi-line
                # payload to have a "data:" prefix.
                # ------------------------------------------------

                normalized = (
                    chunk
                    .replace(
                        "\r\n",
                        "\n"
                    )
                    .replace(
                        "\r",
                        "\n"
                    )
                )

                for line in normalized.split(
                    "\n"
                ):

                    yield (
                        f"data: {line}\n"
                    )

                yield "\n"

            # ------------------------------------------------
            # PRODUCT RESULTS
            # ------------------------------------------------
            #
            # ask_stream() has now completed and the chatbot
            # conversation contains the latest product results.
            #
            # Only send the product event when the product
            # selection changed during this request.
            # ------------------------------------------------

            after_products = build_product_payload(
                chatbot
            )

            after_ids = product_ids(
                after_products
            )

            if (
                after_products
                and after_ids != before_ids
            ):

                product_json = json.dumps(
                    after_products,
                    ensure_ascii=False,
                    default=str
                )

                yield (
                    "event: products\n"
                    f"data: {product_json}\n\n"
                )

            # ------------------------------------------------
            # Streaming completed
            # ------------------------------------------------

            yield (
                "event: done\n"
                "data: [DONE]\n\n"
            )

        except Exception as exc:

            print(
                f"STREAM ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            yield (
                "event: error\n"
                'data: {"message":"Unable to '
                'complete the response."}\n\n'
            )


    # ========================================================
    # STREAMING RESPONSE
    # ========================================================

    return StreamingResponse(
        event_generator(),

        media_type="text/event-stream",

        headers={
            "Cache-Control": (
                "no-cache, no-transform"
            ),

            "Connection": "keep-alive",

            "X-Accel-Buffering": "no",
        }
    )


# ============================================================
# LIVE STORE SYNC
# ============================================================

@app.post("/admin/sync-live-store")
def sync_live_store():
    """Refresh/check the live Zyraluxe catalogue.

    Existing local sessions are cleared so the next session reloads
    the current live product and policy data.
    """
    try:
        products, knowledge = fetch_live_store()
        sessions.clear()

        return {
            "status": "ok",
            "products": len(products),
            "knowledge_documents": len(knowledge),
        }

    except Exception as exc:
        print(
            f"LIVE STORE SYNC ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=502,
            detail="Unable to refresh live Zyraluxe store data."
        ) from exc


# ============================================================
# DELETE SESSION
# ============================================================

@app.delete(
    "/session/{session_id}"
)
def delete_session(
    session_id: str
):

    if session_id not in sessions:

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    del sessions[
        session_id
    ]

    return {
        "status": "deleted",
        "session_id": session_id
    }


# ============================================================
# LOCAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=(
            APP_ENV ==
            "development"
        )
    )
