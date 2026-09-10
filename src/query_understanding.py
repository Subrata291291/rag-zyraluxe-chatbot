import json
import re

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME
)

from src.retry import retry_call

from openai import OpenAI


# ============================================================
# OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)


# ============================================================
# DEFAULT FILTER STRUCTURE
# ============================================================

def empty_filters(query):
    """
    Always return a complete filter dictionary.

    This prevents downstream code from failing when the LLM
    omits a field.
    """

    return {
        "original_query": query,
        "corrected_query": query,

        "category": None,
        "metal": None,
        "material_type": None,
        "karat": None,

        "min_price": None,
        "max_price": None,

        "sort_by": None,
        "sort_order": None
    }


# ============================================================
# TYPO / SPELLING NORMALIZATION
# ============================================================

COMMON_CORRECTIONS = {
    "sliver": "silver",
    "silvr": "silver",
    "silvar": "silver",

    "rigns": "rings",
    "rign": "ring",
    "ringss": "rings",

    "necklase": "necklace",
    "neckless": "necklace",
    "necklesses": "necklaces",

    "earring": "earrings",
    "earringss": "earrings",

    "braclet": "bracelet",
    "bracelett": "bracelet",

    "jewllery": "jewellery",
    "jewelry": "jewellery",
    "jewllery": "jewellery",

    "plattted": "plated",
    "platted": "plated",
    "platd": "plated",
    "platedd": "plated",

    "goldplated": "gold plated",
    "gold-plated": "gold plated",

    "diamon": "diamond",
    "dimond": "diamond",

    "traditonal": "traditional",
    "tradtional": "traditional",

    "modren": "modern",

    "cheepest": "cheapest",
    "cheapestt": "cheapest",

    "expensivee": "expensive"
}


def normalize_typos(text):
    """
    Apply conservative spelling corrections.

    We intentionally correct only known/common mistakes.
    Unknown words are left unchanged.
    """

    corrected = text

    for wrong, right in COMMON_CORRECTIONS.items():

        corrected = re.sub(
            rf"\b{re.escape(wrong)}\b",
            right,
            corrected,
            flags=re.IGNORECASE
        )

    return corrected


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json_object(text):
    """
    Extract the first valid JSON object from an LLM response.

    Handles responses such as:

        {
            "category": "ring"
        }

    and also responses wrapped in markdown:

        ```json
        {
            ...
        }
        ```

    This fixes the previous warning caused by trying to parse
    text containing markdown or extra characters.
    """

    if not text:
        raise ValueError(
            "Query understanding returned empty content."
        )

    text = text.strip()


    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):

            return parsed

    except json.JSONDecodeError:

        pass


    # --------------------------------------------------------
    # Remove markdown fences
    # --------------------------------------------------------

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace(
        "```",
        ""
    ).strip()


    try:

        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):

            return parsed

    except json.JSONDecodeError:

        pass


    # --------------------------------------------------------
    # Find first balanced JSON object
    # --------------------------------------------------------

    start = cleaned.find("{")

    if start == -1:

        raise ValueError(
            "No JSON object found in query-understanding response."
        )


    depth = 0
    in_string = False
    escaped = False


    for index in range(
        start,
        len(cleaned)
    ):

        char = cleaned[index]


        if in_string:

            if escaped:

                escaped = False

            elif char == "\\":

                escaped = True

            elif char == '"':

                in_string = False

            continue


        if char == '"':

            in_string = True

        elif char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                candidate = cleaned[
                    start:index + 1
                ]

                try:

                    parsed = json.loads(
                        candidate
                    )

                    if isinstance(
                        parsed,
                        dict
                    ):

                        return parsed

                except json.JSONDecodeError:

                    break


    raise ValueError(
        "Could not extract valid JSON from "
        "query-understanding response."
    )


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def normalize_nullable(value):
    """
    Convert LLM null-like values into Python None.
    """

    if value is None:
        return None

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return None

        if value.lower() in {
            "none",
            "null",
            "unknown",
            "n/a",
            "na"
        }:

            return None

    return value


def normalize_category(value):
    value = normalize_nullable(value)

    if value is None:
        return None

    value = str(value).lower().strip()

    aliases = {
        "rings": "ring",
        "ring": "ring",

        "necklaces": "necklace",
        "necklace": "necklace",

        "bracelets": "bracelet",
        "bracelet": "bracelet",

        "earrings": "earrings",
        "earring": "earrings"
    }

    return aliases.get(
        value,
        value
    )


def normalize_metal(value):
    value = normalize_nullable(value)

    if value is None:
        return None

    value = str(value).lower().strip()

    aliases = {
        "gold": "gold",
        "yellow gold": "gold",

        "silver": "silver",

        "brass": "brass"
    }

    return aliases.get(
        value,
        value
    )


def normalize_material_type(value):
    value = normalize_nullable(value)

    if value is None:
        return None

    value = str(value).lower().strip()

    value = value.replace(
        "-",
        "_"
    )

    value = value.replace(
        " ",
        "_"
    )

    aliases = {
        "solid_gold": "solid_gold",
        "gold": "solid_gold",

        "gold_plated": "gold_plated",
        "goldplated": "gold_plated",
        "gold_plating": "gold_plated",

        "silver": "silver"
    }

    return aliases.get(
        value,
        value
    )


def normalize_karat(value):
    value = normalize_nullable(value)

    if value is None:
        return None

    value = str(value).strip().upper()

    # "22", "22k", "22 K" -> "22K"
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*K?",
        value
    )

    if match:

        return (
            match.group(1)
            + "K"
        )

    return value


def normalize_price(value):
    value = normalize_nullable(value)

    if value is None:
        return None

    if isinstance(
        value,
        (int, float)
    ):

        return int(value)

    text = str(value).lower().strip()

    text = text.replace(
        "₹",
        ""
    )

    text = text.replace(
        ",",
        ""
    )

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(k|thousand)?",
        text
    )

    if not match:
        return None

    number = float(
        match.group(1)
    )

    suffix = match.group(2)

    if suffix in {
        "k",
        "thousand"
    }:

        number *= 1000

    return int(number)


def normalize_sort_by(value):
    value = normalize_nullable(value)

    if value is None:
        return None

    value = str(value).lower().strip()

    if value in {
        "price",
        "cost"
    }:

        return "price"

    if value in {
        "relevance",
        "semantic",
        "score"
    }:

        return "relevance"

    return None


def normalize_sort_order(value):
    value = normalize_nullable(value)

    if value is None:
        return None

    value = str(value).lower().strip()

    if value in {
        "asc",
        "ascending",
        "low_to_high",
        "lowest"
    }:

        return "asc"

    if value in {
        "desc",
        "descending",
        "high_to_low",
        "highest"
    }:

        return "desc"

    return None


# ============================================================
# PRICE EXTRACTION FALLBACK
# ============================================================

def extract_price_filters(query):
    """
    Deterministic price extraction.

    This is intentionally used as a fallback/correction layer
    because prices should not depend entirely on an LLM.
    """

    q = normalize_typos(
        query.lower()
    )

    min_price = None
    max_price = None


    # --------------------------------------------------------
    # BETWEEN
    # --------------------------------------------------------

    match = re.search(
        r"between\s*₹?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(k|thousand)?\s*and\s*"
        r"₹?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(k|thousand)?",
        q
    )

    if match:

        first = float(
            match.group(1).replace(
                ",",
                ""
            )
        )

        second = float(
            match.group(3).replace(
                ",",
                ""
            )
        )

        suffix1 = match.group(2)
        suffix2 = match.group(4)

        if suffix1 in {
            "k",
            "thousand"
        }:

            first *= 1000

        if suffix2 in {
            "k",
            "thousand"
        }:

            second *= 1000

        min_price = int(
            min(
                first,
                second
            )
        )

        max_price = int(
            max(
                first,
                second
            )
        )

        return min_price, max_price


    # --------------------------------------------------------
    # UNDER / BELOW / UP TO
    # --------------------------------------------------------

    match = re.search(
        r"(?:under|below|less than|up to|upto|max(?:imum)?"
        r"(?:\s+price)?(?:\s+of)?)\s*"
        r"₹?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(k|thousand)?",
        q
    )

    if match:

        number = float(
            match.group(1).replace(
                ",",
                ""
            )
        )

        suffix = match.group(2)

        if suffix in {
            "k",
            "thousand"
        }:

            number *= 1000

        max_price = int(
            number
        )


    # --------------------------------------------------------
    # ABOVE / OVER / MORE THAN
    # --------------------------------------------------------

    match = re.search(
        r"(?:above|over|more than|greater than)\s*"
        r"₹?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(k|thousand)?",
        q
    )

    if match:

        number = float(
            match.group(1).replace(
                ",",
                ""
            )
        )

        suffix = match.group(2)

        if suffix in {
            "k",
            "thousand"
        }:

            number *= 1000

        min_price = int(
            number
        )


    return min_price, max_price


# ============================================================
# SORT DETECTION
# ============================================================

def detect_sort(query):
    q = normalize_typos(
        query.lower()
    )

    if re.search(
        r"\b(cheapest|lowest price|least expensive)\b",
        q
    ):

        return "price", "asc"


    if re.search(
        r"\b(most expensive|highest price|costliest)\b",
        q
    ):

        return "price", "desc"


    return None, None


# ============================================================
# VALIDATE FILTER SEMANTICS
# ============================================================

def validate_filters(
    filters,
    corrected_query
):
    """
    Final deterministic cleanup.

    The LLM proposes the structure, but deterministic rules
    protect important catalog semantics.
    """

    filters["original_query"] = (
        filters.get(
            "original_query"
        )
        or corrected_query
    )

    filters["corrected_query"] = (
        corrected_query
    )


    # --------------------------------------------------------
    # Normalize individual fields
    # --------------------------------------------------------

    filters["category"] = normalize_category(
        filters.get("category")
    )

    filters["metal"] = normalize_metal(
        filters.get("metal")
    )

    filters["material_type"] = normalize_material_type(
        filters.get("material_type")
    )

    filters["karat"] = normalize_karat(
        filters.get("karat")
    )

    filters["min_price"] = normalize_price(
        filters.get("min_price")
    )

    filters["max_price"] = normalize_price(
        filters.get("max_price")
    )

    filters["sort_by"] = normalize_sort_by(
        filters.get("sort_by")
    )

    filters["sort_order"] = normalize_sort_order(
        filters.get("sort_order")
    )


    # --------------------------------------------------------
    # Price fallback
    # --------------------------------------------------------

    extracted_min, extracted_max = (
        extract_price_filters(
            corrected_query
        )
    )


    if extracted_min is not None:

        filters["min_price"] = (
            extracted_min
        )


    if extracted_max is not None:

        filters["max_price"] = (
            extracted_max
        )


    # --------------------------------------------------------
    # Sort fallback
    # --------------------------------------------------------

    sort_by, sort_order = detect_sort(
        corrected_query
    )


    if sort_by:

        filters["sort_by"] = (
            sort_by
        )

        filters["sort_order"] = (
            sort_order
        )


    # --------------------------------------------------------
    # Gold plated semantic protection
    # --------------------------------------------------------
    #
    # If the corrected query clearly contains "gold plated",
    # never allow the LLM to silently turn it into normal gold.
    #

    gold_plated_patterns = [
        r"\bgold\s+plated\b",
        r"\bgold\s+plating\b",
        r"\bgoldplated\b"
    ]


    if any(
        re.search(
            pattern,
            corrected_query.lower()
        )
        for pattern in gold_plated_patterns
    ):

        filters["material_type"] = (
            "gold_plated"
        )

        # "gold plated" describes the material type.
        # It should not become solid gold.
        if filters["metal"] == "gold":

            filters["metal"] = None


    # --------------------------------------------------------
    # Normal gold protection
    # --------------------------------------------------------
    # Only set solid_gold if user explicitly asks for solid/pure gold
    if re.search(r"\b(solid\s+gold|pure\s+gold|real\s+gold)\b", corrected_query.lower()):
        filters["material_type"] = "solid_gold"



    return filters


# ============================================================
# LLM REQUEST
# ============================================================

def _request_query_understanding(
    query
):
    """
    Call the LLM and return raw content.
    """

    system_prompt = """
You are a query-understanding engine for a jewellery
shopping chatbot.

Your job is to understand customer language, including
typos, spelling mistakes, informal wording and shorthand.

Examples:

"sliver rings"
-> "silver rings"

"gold rigns under 20k"
-> "gold rings under 20000"

"gold platted jewellery"
-> "gold plated jewellery"

"necklase below 30k"
-> "necklace below 30000"

"which one is cheapest?"
-> understand that the customer wants price ascending
when it is a product-search request.

IMPORTANT:

"gold" and "gold plated" are NOT the same thing.

gold:
    material_type = "solid_gold"

gold plated:
    material_type = "gold_plated"

Return ONLY one JSON object.

Do not use markdown.
Do not use ```json.
Do not add explanations.
Do not add text before or after the JSON.

The JSON must contain exactly these fields:

{
    "corrected_query": "...",
    "category": null,
    "metal": null,
    "material_type": null,
    "karat": null,
    "min_price": null,
    "max_price": null,
    "sort_by": null,
    "sort_order": null
}

Allowed category values:

ring
necklace
bracelet
earrings
null

Allowed metal values:

gold
silver
brass
null

Allowed material_type values:

solid_gold
gold_plated
silver
null

Allowed sort_by values:

price
relevance
null

Allowed sort_order values:

asc
desc
null

Price examples:

"under 20k"
=> max_price = 20000

"below 30000"
=> max_price = 30000

"above 20000"
=> min_price = 20000

"between 10000 and 30000"
=> min_price = 10000
=> max_price = 30000

Never invent a product.

Never invent a filter that the customer did not request.
"""


    user_prompt = f"""
CUSTOMER QUERY:

{query}

Return only the JSON object.
"""


    def request():

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0,

            timeout=30,

            max_tokens=300
        )


        if not response.choices:

            raise RuntimeError(
                "OpenRouter returned no choices."
            )


        message = response.choices[0].message


        if message is None:

            raise RuntimeError(
                "OpenRouter returned an empty message."
            )


        content = message.content


        if not content:

            raise RuntimeError(
                "OpenRouter returned empty content."
            )


        return content.strip()


    return retry_call(
        request,
        max_retries=3,
        base_delay=1
    )


def extract_fast_filters(query):
    """
    Extract structured filters deterministically using fast keyword/regex rules.
    Returns (filters_dict, is_confident).
    """
    q = query.lower()
    filters = empty_filters(query)
    found_any = False

    # 1. Category
    category_patterns = [
        (r"\b(ring|rings)\b", "ring"),
        (r"\b(necklace|necklaces|chokker|choker|chain|chains)\b", "necklace"),
        (r"\b(earring|earrings|jhumka|jhumki)\b", "earrings"),
        (r"\b(bracelet|bracelets|bangle|bangles)\b", "bracelet"),
        (r"\b(pendant|pendants|pendent|pendents)\b", "necklace"),
        (r"\b(anklet|anklets)\b", "anklets"),
    ]
    for pattern, canonical in category_patterns:
        if re.search(pattern, q):
            filters["category"] = canonical
            found_any = True
            break

    # 2. Metal & Material type
    if re.search(r"\b(gold\s+plated|goldplated|gold\s+plating)\b", q):
        filters["material_type"] = "gold_plated"
        found_any = True
    elif re.search(r"\b(solid\s+gold|pure\s+gold|real\s+gold)\b", q):
        filters["metal"] = "gold"
        filters["material_type"] = "solid_gold"
        found_any = True
    elif re.search(r"\bgold\b", q):
        filters["metal"] = "gold"
        found_any = True

    if re.search(r"\bsilver\b", q):
        filters["metal"] = "silver"
        found_any = True

    if re.search(r"\bbrass\b", q):
        filters["metal"] = "brass"
        found_any = True

    # 3. Karat
    karat_match = re.search(r"\b(9|14|18|22|24)\s*k(?:arat)?\b", q)
    if karat_match:
        filters["karat"] = f"{karat_match.group(1)}K"
        found_any = True

    # 4. Price
    min_p, max_p = extract_price_filters(q)
    if min_p is not None or max_p is not None:
        filters["min_price"] = min_p
        filters["max_price"] = max_p
        found_any = True

    # 5. Sorting
    sort_by, sort_order = detect_sort(q)
    if sort_by:
        filters["sort_by"] = sort_by
        filters["sort_order"] = sort_order
        found_any = True

    return filters, found_any


# ============================================================
# PUBLIC QUERY UNDERSTANDING FUNCTION
# ============================================================

def understand_query(
    query
):
    """
    Convert natural-language product requests into
    deterministic structured filters.

    Optimized for speed:
        1. Correct obvious typos locally.
        2. Try fast deterministic extraction (0ms).
        3. Only if no structured filters are recognized, query LLM.
        4. Validate and normalize.
    """

    query = (
        query
        .strip()
    )


    if not query:

        return empty_filters(
            query
        )


    # --------------------------------------------------------
    # STEP 1: TYPO CORRECTION
    # --------------------------------------------------------

    corrected_query = normalize_typos(
        query
    )


    # --------------------------------------------------------
    # STEP 2: FAST DETERMINISTIC EXTRACTION
    # --------------------------------------------------------
    fast_filters, has_fast_matches = extract_fast_filters(corrected_query)
    if has_fast_matches:
        fast_filters["original_query"] = query
        fast_filters["corrected_query"] = corrected_query
        return validate_filters(fast_filters, corrected_query)

    # --------------------------------------------------------
    # STEP 3: DEFAULT STRUCTURE & LLM UNDERSTANDING FALLBACK
    # --------------------------------------------------------

    filters = empty_filters(
        query
    )


    filters["corrected_query"] = (
        corrected_query
    )


    try:
        raw_response = (
            _request_query_understanding(
                corrected_query
            )
        )


        print()
        print("===== RAW QUERY UNDERSTANDING RESPONSE =====")
        print(repr(raw_response))

        # ----------------------------------------------------
        # STEP 4: SAFE JSON EXTRACTION
        # ----------------------------------------------------
        parsed = extract_json_object(
            raw_response
        )

        if isinstance(
            parsed,
            dict
        ):
            filters.update(
                parsed
            )

    except Exception as error:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We do NOT crash the product search because the LLM
        # returned malformed JSON or temporarily failed.
        #
        # Deterministic extraction below can still recover
        # price/sort information and the corrected query.
        # ----------------------------------------------------

        print()

        print(
            "Query understanding warning:"
        )

        print(
            str(error)
        )

        print(
            "Using deterministic query normalization "
            "fallback."
        )


    # --------------------------------------------------------
    # STEP 5: FINAL VALIDATION
    # --------------------------------------------------------

    filters = validate_filters(

        filters,

        corrected_query

    )

    print()
    print("===== FINAL FILTERS =====")
    print(filters)

    return filters
