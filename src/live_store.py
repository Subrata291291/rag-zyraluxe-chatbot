"""Live Zyraluxe WooCommerce/WordPress data loader.

Uses WooCommerce's public Store API for published products and fetches
customer-facing policy/content pages from the live Zyraluxe website.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


SITE_URL = os.getenv("ZYRALUXE_SITE_URL", "https://zyraluxe.in").rstrip("/")
STORE_API_URL = os.getenv(
    "ZYRALUXE_STORE_API_URL",
    f"{SITE_URL}/wp-json/wc/store/v1/products",
)

REQUEST_TIMEOUT = int(os.getenv("LIVE_STORE_TIMEOUT", "20"))
CACHE_TTL = int(os.getenv("LIVE_STORE_CACHE_TTL", "86400"))  # 24 hours default
PER_PAGE = 100

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_FILE = _CACHE_DIR / "live_cache.json"

_CACHED_PRODUCTS: list[dict[str, Any]] | None = None
_CACHED_KNOWLEDGE: list[dict[str, str]] | None = None
_LAST_FETCH_TIME: float = 0.0
PER_PAGE = 100

HEADERS = {
    "User-Agent": "Zyraluxe-AI-Assistant/1.0 (+https://zyraluxe.in)",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
}

# Explicit pages that are known to contain customer-facing store information.
# Discovery below can add more policy/help pages found from the homepage.
KNOWN_KNOWLEDGE_URLS = [
    f"{SITE_URL}/",
    f"{SITE_URL}/return-policy/",
    f"{SITE_URL}/privacy-policy/",
]

DISCOVERY_WORDS = (
    "policy", "policies", "return", "refund", "shipping", "delivery",
    "privacy", "terms", "condition", "about", "contact", "faq", "help",
    "payment", "cancellation",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_category(categories: list[dict[str, Any]], name: str) -> str:
    text = " ".join(
        [name] + [str(c.get("name", "")) for c in categories]
    ).lower()

    if "earring" in text:
        return "earrings"
    if "necklace" in text:
        return "necklace"
    if "bracelet" in text:
        return "bracelet"
    if "bangle" in text:
        return "bangles"
    if "jhumka" in text or "jhumki" in text:
        return "oxidised jhumka"
    if "chokker" in text or "choker" in text:
        return "chokker"
    if "pendent" in text or "pendant" in text:
        return "pendents"
    if "anklet" in text:
        return "anklets"
    if "sitahar" in text:
        return "sitahar"
    if "combo" in text:
        return "combo"
    if "ring" in text:
        return "ring"
    return (categories[0].get("name") if categories else "jewellery") or "jewellery"


def _infer_metal(name: str, description: str, categories: list[dict[str, Any]]) -> str:
    text = " ".join(
        [name, description] + [str(c.get("name", "")) for c in categories]
    ).lower()
    if "gold" in text:
        return "gold"
    if "silver" in text:
        return "silver"
    return "not specified"


def _infer_karat(name: str, description: str) -> str:
    text = f"{name} {description}"
    match = re.search(r"\b(9|14|18|22|24)\s*k(?:arat)?\b", text, flags=re.I)
    return f"{match.group(1)}K" if match else "Not specified"


def _price_from_product(product: dict[str, Any]) -> int:
    prices = product.get("prices") or {}
    raw = prices.get("price")
    if raw not in (None, ""):
        try:
            minor = int(str(raw))
            precision = int(prices.get("currency_minor_unit", 2) or 2)
            return int(round(minor / (10 ** precision)))
        except (TypeError, ValueError):
            pass

    # Fallback for older/custom Store API responses.
    for key in ("price", "regular_price", "sale_price"):
        value = product.get(key)
        if value not in (None, ""):
            try:
                return int(float(str(value).replace(",", "")))
            except ValueError:
                continue

    price_range = product.get("prices", {}).get("price_range") or {}
    min_price = price_range.get("min_amount")
    if min_price:
        try:
            precision = int(prices.get("currency_minor_unit", 2) or 2)
            return int(round(int(min_price) / (10 ** precision)))
        except (TypeError, ValueError):
            pass

    return 0


def _normalize_product(product: dict[str, Any]) -> dict[str, Any]:
    name = _clean_text(product.get("name"))
    description = _clean_text(
        product.get("description") or product.get("short_description") or product.get("summary")
    )
    categories = product.get("categories") or []
    category = _canonical_category(categories, name)
    metal = _infer_metal(name, description, categories)
    karat = _infer_karat(name, description)
    price = _price_from_product(product)

    images = product.get("images") or []
    image_url = ""
    if images:
        image_url = str(images[0].get("src") or images[0].get("thumbnail") or "")

    return {
        # Keep the existing chatbot schema compatible.
        "id": str(product.get("id")),
        "name": name,
        "category": category,
        "metal": metal,
        "material_type": "fashion_jewellery",
        "karat": karat,
        "price": price,
        "description": description,

        # Live WooCommerce fields for UI / grounding.
        "sku": str(product.get("sku") or ""),
        "slug": str(product.get("slug") or ""),
        "url": str(product.get("permalink") or ""),
        "image_url": image_url,
        "stock_status": "instock" if product.get("is_in_stock") else "outofstock",
        "is_in_stock": bool(product.get("is_in_stock")),
        "stock_quantity": product.get("low_stock_remaining"),
        "on_sale": bool(product.get("on_sale")),
        "rating": product.get("average_rating"),
        "review_count": product.get("review_count"),
        "attributes": product.get("attributes") or [],
        "tags": product.get("tags") or [],
        "raw_categories": categories,
    }


def fetch_live_products() -> list[dict[str, Any]]:
    """Fetch all published products from WooCommerce Store API."""
    products: list[dict[str, Any]] = []

    for page in range(1, 101):
        response = requests.get(
            STORE_API_URL,
            params={"page": page, "per_page": PER_PAGE},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break
        products.extend(_normalize_product(item) for item in batch)

        total_pages = response.headers.get("X-WP-TotalPages")
        if total_pages and page >= int(total_pages):
            break
        if len(batch) < PER_PAGE:
            break

    if not products:
        raise RuntimeError("The live WooCommerce Store API returned no products.")

    return products


def _discover_knowledge_urls() -> list[str]:
    urls = list(KNOWN_KNOWLEDGE_URLS)
    try:
        response = requests.get(SITE_URL + "/", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = urljoin(SITE_URL + "/", link.get("href"))
            text = _clean_text(link.get_text(" ", strip=True))
            parsed = urlparse(href)
            if parsed.netloc not in {urlparse(SITE_URL).netloc, f"www.{urlparse(SITE_URL).netloc}"}:
                continue
            haystack = f"{text} {href}".lower()
            if any(word in haystack for word in DISCOVERY_WORDS):
                clean = href.split("#", 1)[0]
                if clean not in urls:
                    urls.append(clean)
    except Exception as exc:
        print(f"LIVE KNOWLEDGE DISCOVERY WARNING: {exc}")
    return urls


def fetch_live_knowledge() -> list[dict[str, str]]:
    """Fetch customer-facing policy/help content from the live site."""
    documents: list[dict[str, str]] = []

    for url in _discover_knowledge_urls():
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()

            title = _clean_text(soup.title.get_text()) if soup.title else url
            main = soup.find("main") or soup.find("article") or soup.body
            content = _clean_text(main.get_text(" ", strip=True) if main else response.text)

            if not content or len(content) < 80:
                continue

            documents.append({
                "document": title,
                "content": content,
                "url": url,
            })
        except Exception as exc:
            print(f"LIVE KNOWLEDGE WARNING [{url}]: {exc}")

    return documents


def _load_disk_cache() -> tuple[list[dict[str, Any]], list[dict[str, str]]] | None:
    """Try to load products and knowledge from disk cache."""
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            products = data.get("products")
            knowledge = data.get("knowledge")
            saved_at = data.get("saved_at", 0)
            if isinstance(products, list) and isinstance(knowledge, list):
                # Valid if within TTL or as fallback
                return products, knowledge
    except Exception as exc:
        print(f"CACHE LOAD WARNING: {exc}")
    return None


def _save_disk_cache(products: list[dict[str, Any]], knowledge: list[dict[str, str]]) -> None:
    """Save products and knowledge to disk cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "saved_at": time.time(),
                "products": products,
                "knowledge": knowledge,
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(products)} products and {len(knowledge)} knowledge docs to cache.")
    except Exception as exc:
        print(f"CACHE SAVE WARNING: {exc}")


def fetch_live_store(force_refresh: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Fetch live products and knowledge with in-memory and disk caching."""
    global _CACHED_PRODUCTS, _CACHED_KNOWLEDGE, _LAST_FETCH_TIME

    now = time.time()

    # 1. In-memory cache hit
    if not force_refresh and _CACHED_PRODUCTS is not None and _CACHED_KNOWLEDGE is not None:
        if now - _LAST_FETCH_TIME < CACHE_TTL:
            return _CACHED_PRODUCTS, _CACHED_KNOWLEDGE

    # 2. Disk cache hit
    if not force_refresh:
        disk_data = _load_disk_cache()
        if disk_data is not None:
            _CACHED_PRODUCTS, _CACHED_KNOWLEDGE = disk_data
            _LAST_FETCH_TIME = now
            print(f"Loaded store data from cache: {len(_CACHED_PRODUCTS)} products, {len(_CACHED_KNOWLEDGE)} knowledge docs.")
            return _CACHED_PRODUCTS, _CACHED_KNOWLEDGE

    # 3. Network fetch (first run or force refresh)
    print("Fetching live store data from Zyraluxe website...")
    try:
        products = fetch_live_products()
        knowledge = fetch_live_knowledge()
        _CACHED_PRODUCTS = products
        _CACHED_KNOWLEDGE = knowledge
        _LAST_FETCH_TIME = now
        _save_disk_cache(products, knowledge)
        print(f"Live Zyraluxe products loaded: {len(products)}")
        print(f"Live Zyraluxe knowledge pages loaded: {len(knowledge)}")
        return products, knowledge
    except Exception as exc:
        print(f"LIVE STORE FETCH FAILED: {exc}")
        # Try fallback to disk cache even if expired
        disk_data = _load_disk_cache()
        if disk_data is not None:
            print("Using stale disk cache as fallback.")
            _CACHED_PRODUCTS, _CACHED_KNOWLEDGE = disk_data
            return _CACHED_PRODUCTS, _CACHED_KNOWLEDGE
        raise

