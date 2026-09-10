from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME
)

from src.prompt_loader import load_prompts
from src.retry import retry_call


# ============================================================
# OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)


# ============================================================
# LOAD PROMPTS
# ============================================================

PROMPTS = load_prompts()


# ============================================================
# NORMAL LLM CALL
# ============================================================

def call_llm(messages):

    def request():

        return client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            timeout=30,
            max_tokens=600
        )

    return retry_call(
        request,
        max_retries=3,
        base_delay=1
    )


# ============================================================
# STREAMING LLM CALL
# ============================================================

def stream_llm(messages):

    """
    Stream the response from OpenRouter.

    Yields only valid text content.
    Safely handles empty choices, empty deltas
    and empty content.
    """

    def request():

        return client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            timeout=30,
            max_tokens=600,
            stream=True
        )

    stream = retry_call(
        request,
        max_retries=3,
        base_delay=1
    )

    if stream is None:
        raise RuntimeError(
            "OpenRouter returned no streaming response."
        )

    for chunk in stream:

        if chunk is None:
            continue

        choices = getattr(
            chunk,
            "choices",
            None
        )

        if not choices:
            continue

        choice = choices[0]

        if choice is None:
            continue

        delta = getattr(
            choice,
            "delta",
            None
        )

        if delta is None:
            continue

        content = getattr(
            delta,
            "content",
            None
        )

        if content:
            yield content


# ============================================================
# GENERIC NON-STREAMING GENERATOR
# ============================================================

def generate_with_prompt(
    system_prompt,
    user_prompt
):

    response = call_llm(
        [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    if response is None:
        raise RuntimeError(
            "OpenRouter returned no response."
        )

    choices = getattr(
        response,
        "choices",
        None
    )

    if not choices:
        raise RuntimeError(
            "OpenRouter returned no choices."
        )

    choice = choices[0]

    if choice is None:
        raise RuntimeError(
            "OpenRouter returned an empty choice."
        )

    message = getattr(
        choice,
        "message",
        None
    )

    if message is None:
        raise RuntimeError(
            "OpenRouter returned an empty message."
        )

    answer = getattr(
        message,
        "content",
        None
    )

    if not answer:
        raise RuntimeError(
            "OpenRouter returned empty content."
        )

    return answer.strip()


# ============================================================
# GENERIC STREAMING GENERATOR
# ============================================================

def generate_stream_with_prompt(
    system_prompt,
    user_prompt
):

    """
    Generate a response progressively.

    Yields text chunks as they arrive.
    """

    if not system_prompt:
        raise RuntimeError(
            "System prompt is empty."
        )

    if not user_prompt:
        raise RuntimeError(
            "User prompt is empty."
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    for chunk in stream_llm(messages):

        if chunk:
            yield chunk


# ============================================================
# PRODUCT ANSWER
# ============================================================

def generate_product_answer(
    query,
    context
):

    """
    Generate a product answer from an already-built
    product context string.

    The caller is responsible for:

    - query understanding
    - filtering
    - semantic search
    - building product context

    This function only generates the answer.
    """

    if not query:
        raise ValueError(
            "Product query is empty."
        )

    if not context:
        raise ValueError(
            "Product context is empty."
        )

    prompt = PROMPTS["product"].format(
        query=query,
        context=context
    )

    response = call_llm(
        [
            {
                "role": "system",
                "content": (
                    "You are a strictly grounded "
                    "jewellery shopping assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    if response is None:
        raise RuntimeError(
            "OpenRouter returned no product response."
        )

    choices = getattr(
        response,
        "choices",
        None
    )

    if not choices:
        raise RuntimeError(
            "OpenRouter returned no product choices."
        )

    choice = choices[0]

    if choice is None:
        raise RuntimeError(
            "OpenRouter returned an empty product choice."
        )

    message = getattr(
        choice,
        "message",
        None
    )

    if message is None:
        raise RuntimeError(
            "OpenRouter returned an empty product message."
        )

    answer = getattr(
        message,
        "content",
        None
    )

    if not answer:
        raise RuntimeError(
            "OpenRouter returned empty product content."
        )

    return answer.strip()


# ============================================================
# PRODUCT STREAM
# ============================================================

def stream_product_answer(
    query,
    context
):

    """
    Stream a product answer using only the
    retrieved product context.
    """

    if not query:
        raise ValueError(
            "Product query is empty."
        )

    if not context:
        raise ValueError(
            "Product context is empty."
        )

    product_prompt = PROMPTS.get(
        "product"
    )

    if not product_prompt:
        raise RuntimeError(
            "Product prompt is missing."
        )

    system_prompt = PROMPTS.get(
        "system"
    )

    if not system_prompt:
        raise RuntimeError(
            "System prompt is missing."
        )

    prompt = product_prompt.format(
        query=query,
        context=context
    )

    yield from generate_stream_with_prompt(
        system_prompt,
        prompt
    )


# ============================================================
# KNOWLEDGE ANSWER
# ============================================================

def generate_knowledge_answer(
    query,
    context
):

    if not query:
        raise ValueError(
            "Knowledge query is empty."
        )

    if not context:
        raise ValueError(
            "Knowledge context is empty."
        )

    prompt = PROMPTS["knowledge"].format(
        query=query,
        context=context
    )

    return generate_with_prompt(
        PROMPTS["system"],
        prompt
    )


# ============================================================
# KNOWLEDGE STREAM
# ============================================================

def stream_knowledge_answer(
    query,
    context
):

    if not query:
        raise ValueError(
            "Knowledge query is empty."
        )

    if not context:
        raise ValueError(
            "Knowledge context is empty."
        )

    prompt = PROMPTS["knowledge"].format(
        query=query,
        context=context
    )

    yield from generate_stream_with_prompt(
        PROMPTS["system"],
        prompt
    )


# ============================================================
# FOLLOW-UP ANSWER
# ============================================================

def generate_followup_answer(
    query,
    context,
    history
):

    if not query:
        raise ValueError(
            "Follow-up query is empty."
        )

    prompt = PROMPTS["followup"].format(
        query=query,
        context=context,
        history=history
    )

    return generate_with_prompt(
        PROMPTS["system"],
        prompt
    )


# ============================================================
# FOLLOW-UP STREAM
# ============================================================

def stream_followup_answer(
    query,
    context,
    history
):

    if not query:
        raise ValueError(
            "Follow-up query is empty."
        )

    prompt = PROMPTS["followup"].format(
        query=query,
        context=context,
        history=history
    )

    yield from generate_stream_with_prompt(
        PROMPTS["system"],
        prompt
    )


# ============================================================
# NORMAL CONVERSATION ANSWER
# ============================================================

def generate_normal_answer(
    query,
    history
):

    if not query:
        raise ValueError(
            "Normal query is empty."
        )

    prompt = PROMPTS["normal"].format(
        query=query,
        history=history
    )

    return generate_with_prompt(
        PROMPTS["system"],
        prompt
    )


# ============================================================
# NORMAL CONVERSATION STREAM
# ============================================================

def stream_normal_answer(
    query,
    history
):

    if not query:
        raise ValueError(
            "Normal query is empty."
        )

    prompt = PROMPTS["normal"].format(
        query=query,
        history=history
    )

    yield from generate_stream_with_prompt(
        PROMPTS["system"],
        prompt
    )