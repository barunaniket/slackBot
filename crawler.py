import os
import re
import time
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from typing import Set, List, Dict

# --- URL NORMALIZATION ---
def normalize_url(url: str) -> str:
    """Normalize URL to avoid duplicates."""
    parsed = urlparse(url)
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip('/') if parsed.path != '/' else parsed.path,
        parsed.params,
        parsed.query,
        ''
    ))
    return normalized

def should_crawl_url(url: str, base_domain: str, start_paths: List[str]) -> bool:
    """Determine if a URL should be crawled with support for multiple documentation paths."""
    parsed = urlparse(url)
    
    if parsed.netloc != base_domain:
        return False
    
    # Check if the URL path starts with any of the allowed paths
    path_allowed = any(parsed.path.startswith(path) for path in start_paths)
    if not path_allowed:
        return False
    
    skip_patterns = [
        r'/search', r'/login', r'/logout', r'/api/', r'/download/',
        r'\.pdf$', r'\.zip$', r'\.exe$', r'\.dmg$', r'\.jpg$', r'\.png$',
        r'/print/', r'/share/', r'/export/', r'#'
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, parsed.path, re.IGNORECASE):
            return False
    
    return True

# --- CONTENT EXTRACTION & CHUNKING ---
def extract_main_content(soup: BeautifulSoup, url: str) -> Dict[str, str]:
    """Enhanced content extraction for documentation sites."""
    title = ""
    if soup.title:
        title = soup.title.string.strip() if soup.title.string else ""
    elif soup.find('h1'):
        title = soup.find('h1').get_text(strip=True)
    
    # More comprehensive selectors for documentation sites
    main_selectors = [
        "main", "article", "[role='main']", "#main-content", 
        ".main-content", ".content", ".article-body", ".documentation",
        ".doc-content", ".post-content", "#content", ".page-content",
        ".docs-content", ".markdown-body", ".md-content", ".guide-content"
    ]
    
    content_element = None
    for selector in main_selectors:
        content_element = soup.select_one(selector)
        if content_element:
            break
    
    if not content_element:
        content_element = soup.body
    
    if not content_element:
        return {"title": title, "content": ""}
    
    # Remove navigation, sidebars, etc.
    for element in content_element(["script", "style", "nav", "footer", "aside", 
                                     "form", "header", "iframe", "noscript", 
                                     ".navigation", ".sidebar", ".ad", ".advertisement",
                                     ".breadcrumb", ".toc", ".table-of-contents"]):
        element.decompose()
    
    # Handle code blocks specially
    for code in content_element.find_all(['pre', 'code']):
        # Preserve code formatting
        code['data-code'] = True
    
    text = content_element.get_text(separator='\n', strip=True)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return {"title": title, "content": text}

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Chunks text into smaller pieces."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(para) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
        else:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# --- CRAWL STATE MANAGEMENT ---
def save_crawl_state(state_file: str, visited_urls: Set[str], to_visit: List[str], 
                    stats: Dict, processed_urls: Set[str]):
    """Save the current crawl state to a file with processed URLs tracking."""
    state = {
        "visited_urls": list(visited_urls),
        "to_visit": to_visit,
        "stats": stats,
        "processed_urls": list(processed_urls),
        "timestamp": time.time()
    }
    with open(state_file, 'w') as f:
        json.dump(state, f)

def load_crawl_state(state_file: str) -> Dict:
    """Load the crawl state from a file."""
    if not os.path.exists(state_file):
        return {
            "visited_urls": set(),
            "to_visit": [],
            "stats": defaultdict(int),
            "processed_urls": set()
        }
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
            return {
                "visited_urls": set(state.get("visited_urls", [])),
                "to_visit": state.get("to_visit", []),
                "stats": defaultdict(int, state.get("stats", {})),
                "processed_urls": set(state.get("processed_urls", []))
            }
    except Exception as e:
        print(f"Error loading crawl state: {e}")
        return {
            "visited_urls": set(),
            "to_visit": [],
            "stats": defaultdict(int),
            "processed_urls": set()
        }

# --- HTML-BASED CRAWLER FOR ULTRALYTICS DOCS ---
def crawl_ultralytics_docs(
    start_url: str,
    max_pages: int = None,
    output_dir: str = "slackBot/support_db/data/"
):
    """Fast HTML-based crawler for Ultralytics documentation."""
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc
    start_path = parsed_start.path if parsed_start.path else "/"
    allowed_paths = ["/"]  # crawl all internal docs and subpaths

    # State
    visited_urls: Set[str] = set()
    to_visit: List[str] = [start_url]
    stats = {"scraped": 0, "skipped": 0, "errors": 0, "links_found": 0}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n🌐 Starting Ultralytics docs crawl: {start_url}")
    print(f"   Output directory: {output_dir}")
    print(f"   Max pages: {max_pages or 'unlimited'}")

    binary_exts = (".pdf", ".zip", ".exe", ".dmg", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".mp4", ".mp3", ".mov")

    while to_visit and (max_pages is None or stats["scraped"] < max_pages):
        url = to_visit.pop(0)
        normalized_url = normalize_url(url)
        if normalized_url in visited_urls:
            continue
        visited_urls.add(normalized_url)

        # Skip binary/irrelevant links
        if any(normalized_url.lower().endswith(ext) for ext in binary_exts):
            stats["skipped"] += 1
            print(f"⊘ Skipped (binary): {normalized_url}")
            continue
        if "#" in normalized_url:
            stats["skipped"] += 1
            print(f"⊘ Skipped (fragment): {normalized_url}")
            continue

        # Only crawl internal docs pages
        if not should_crawl_url(normalized_url, base_domain, allowed_paths):
            stats["skipped"] += 1
            print(f"⊘ Skipped (external or disallowed): {normalized_url}")
            continue

        print(f"[{stats['scraped']+1}] Crawling: {normalized_url}")
        try:
            resp = requests.get(normalized_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
            final_url = resp.url
            if final_url != normalized_url:
                normalized_url = normalize_url(final_url)
            if not resp.ok or "text/html" not in resp.headers.get("Content-Type", ""):
                stats["skipped"] += 1
                print(f"   ⊘ Skipped (not HTML or bad response): {normalized_url} (Status: {resp.status_code})")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            extracted = extract_main_content(soup, normalized_url)
            if extracted["content"] and len(extracted["content"]) > 100:
                # Save as .txt file
                safe_path = (
                    normalized_url.replace("https://", "")
                    .replace("http://", "")
                    .replace("/", "_")
                    .replace("?", "_")
                    .replace("#", "_")
                )
                out_path = os.path.join(output_dir, safe_path + ".txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(normalized_url + "\n")
                    f.write(extracted["content"])
                stats["scraped"] += 1
                print(f"   ✓ Saved: {out_path}")
            else:
                stats["skipped"] += 1
                print(f"   ⊘ Skipped (no substantial content): {normalized_url}")

            # Find and queue new links
            links_found = 0
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("mailto:") or href.startswith("tel:"):
                    continue
                absolute_url = urljoin(normalized_url, href)
                absolute_url = normalize_url(absolute_url)
                if (
                    should_crawl_url(absolute_url, base_domain, allowed_paths)
                    and absolute_url not in visited_urls
                    and absolute_url not in to_visit
                    and not any(absolute_url.lower().endswith(ext) for ext in binary_exts)
                    and "#" not in absolute_url
                ):
                    to_visit.append(absolute_url)
                    links_found += 1
            stats["links_found"] += links_found
            if links_found > 0:
                print(f"   → Found {links_found} new links (Queue: {len(to_visit)})")
            if stats["scraped"] % 10 == 0 and stats["scraped"] > 0:
                print(f"📊 Progress: {stats['scraped']} pages saved, {len(visited_urls)} visited, {len(to_visit)} queued")
        except Exception as e:
            stats["errors"] += 1
            print(f"   ✗ Error: {e}")

    print(f"\n✅ Crawling complete!")
    print(f"   Pages visited: {len(visited_urls)}")
    print(f"   Pages saved: {stats['scraped']}")
    print(f"   Links found: {stats['links_found']}")
    print(f"   Errors: {stats['errors']}")
    print(f"   Skipped: {stats['skipped']}")

if __name__ == "__main__":
    print("=" * 60)
    print("Ultralytics Documentation HTML Crawler")
    print("=" * 60)
    start_url = "https://docs.ultralytics.com/"
    max_pages = None  # Set to None for unlimited
    output_dir = "slackBot/support_db/data/"
    crawl_ultralytics_docs(start_url, max_pages=max_pages, output_dir=output_dir)