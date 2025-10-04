import asyncio
import os
import re
import lancedb
import openai
import json
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from dotenv import load_dotenv
from typing import Set, List, Dict
from collections import defaultdict

# Check if playwright is available
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright not installed. Install with: pip install playwright && playwright install chromium")

# --- CONFIGURATION & INITIALIZATION ---
print("Loading configuration...")
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
db = lancedb.connect("./support_db")

class SupportDoc(lancedb.pydantic.LanceModel):
    text: str
    vector: lancedb.pydantic.Vector(1536)
    source_url: str
    chunk_ref: str = ""  # Reference to specific chunk
    page_title: str = ""
    crawled_at: str = ""

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

# --- PLAYWRIGHT CRAWLER (for JS-heavy sites) ---
async def crawl_with_playwright(start_url: str, max_pages: int = None, table=None, 
                               state_file: str = None, additional_paths: List[str] = None):
    """Enhanced crawler with support for multiple documentation paths."""
    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright is not installed. Please run:")
        print("   pip install playwright")
        print("   playwright install chromium")
        return []
    
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc
    
    # Extract the initial path and add any additional paths
    path_parts = [p for p in parsed_start.path.split('/') if p]
    start_path = '/' + path_parts[0] if path_parts else '/'
    
    # For Zscaler and similar sites, add common documentation paths
    default_paths = [start_path]
    if additional_paths:
        default_paths.extend(additional_paths)
    
    # For Zscaler specifically, add these paths
    if "zscaler.com" in base_domain:
        default_paths.extend(["/zia", "/zpa", "/zdx", "/zda", "/zcc", "/ztna"])
    
    # Load previous state if available
    state = load_crawl_state(state_file) if state_file else {
        "visited_urls": set(),
        "to_visit": [],
        "stats": defaultdict(int),
        "processed_urls": set()
    }
    
    visited_urls = state["visited_urls"]
    to_visit = state["to_visit"] if state["to_visit"] else [start_url]
    stats = state["stats"]
    processed_urls = state["processed_urls"]
    
    print(f"\n🌐 Using Playwright (JavaScript-enabled browser)")
    print(f"   Base domain: {base_domain}")
    print(f"   Start paths: {', '.join(default_paths)}")
    print(f"   Max pages: {max_pages or 'unlimited'}")
    print(f"   Resuming from previous crawl: {len(visited_urls)} URLs already visited")
    print(f"   URLs in queue: {len(to_visit)}")
    print(f"   URLs already processed: {len(processed_urls)}\n")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        # Counter for batch processing
        batch_counter = 0
        save_interval = 5  # Save state every 5 pages
        
        while to_visit and (max_pages is None or len(processed_urls) < max_pages):
            url = to_visit.pop(0)
            normalized_url = normalize_url(url)
            
            if normalized_url in visited_urls:
                continue
            
            visited_urls.add(normalized_url)
            print(f"[{len(processed_urls)+1}] Scraping: {normalized_url}")
            
            try:
                # Navigate and wait for content
                await page.goto(normalized_url, wait_until='networkidle', timeout=30000)
                await asyncio.sleep(1)  # Extra wait for JS to render
                
                # Get rendered HTML
                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract content
                extracted = extract_main_content(soup, normalized_url)
                
                if extracted['content'] and len(extracted['content']) > 100:
                    # Process and ingest this page immediately
                    await process_and_ingest_page({
                        "url": normalized_url,
                        "content": extracted['content'],
                        "title": extracted['title']
                    }, table)
                    
                    processed_urls.add(normalized_url)
                    stats['scraped'] += 1
                    print(f"   ✓ Scraped and saved: {extracted['title'][:60]}...")
                else:
                    print(f"   ⊘ Skipped: No substantial content")
                
                # Find links
                links_found = 0
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(normalized_url, href)
                    absolute_url = normalize_url(absolute_url)
                    
                    if (should_crawl_url(absolute_url, base_domain, default_paths) 
                        and absolute_url not in visited_urls
                        and absolute_url not in to_visit):
                        to_visit.append(absolute_url)
                        links_found += 1
                
                stats['links_found'] += links_found
                
                if links_found > 0:
                    print(f"   → Found {links_found} new links (Queue: {len(to_visit)})")
                
                # Save state periodically
                batch_counter += 1
                if batch_counter % save_interval == 0:
                    save_crawl_state(state_file, visited_urls, to_visit, stats, processed_urls)
                    print(f"   💾 Saved crawl state (visited {len(visited_urls)} URLs, processed {len(processed_urls)})")
                
                # Progress update every 10 pages
                if len(processed_urls) % 10 == 0:
                    print(f"\n📊 Progress: {stats['scraped']} pages scraped, "
                          f"{len(visited_urls)} visited, {len(to_visit)} queued\n")
                
            except PlaywrightTimeout:
                stats['errors'] += 1
                print(f"   ✗ Timeout on {normalized_url}")
            except Exception as e:
                stats['errors'] += 1
                print(f"   ✗ Error: {e}")
            
            await asyncio.sleep(0.5)  # Be polite
        
        # Final state save
        save_crawl_state(state_file, visited_urls, to_visit, stats, processed_urls)
        await browser.close()
    
    print(f"\n✅ Crawling complete!")
    print(f"   Pages visited: {len(visited_urls)}")
    print(f"   Pages scraped: {stats['scraped']}")
    print(f"   Links found: {stats['links_found']}")
    print(f"   Errors: {stats['errors']}")
    
    return stats['scraped']

# --- PAGE PROCESSING AND INGESTION ---
async def process_and_ingest_page(page: Dict, table):
    """Process a single page and ingest it into the database with URL tracking."""
    chunks = chunk_text(page['content'], max_chars=1000)
    
    if not chunks:
        return
    
    # Create documents for this page
    docs = []
    for i, text_chunk in enumerate(chunks):
        if text_chunk.strip():
            # Add chunk number for reference
            chunk_ref = f"{page['url']}#chunk-{i+1}"
            docs.append({
                "text": text_chunk,
                "source_url": page['url'],
                "chunk_ref": chunk_ref,
                "page_title": page.get('title', ''),
                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })
    
    if not docs:
        return
    
    # Create embeddings
    texts = [doc["text"] for doc in docs]
    try:
        response = await asyncio.to_thread(
            openai.embeddings.create, 
            input=texts, 
            model="text-embedding-ada-002"
        )
        embeddings = [item.embedding for item in response.data]
        
        # Prepare data for insertion
        data_to_add = [
            {
                "text": doc["text"], 
                "vector": emb, 
                "source_url": doc["source_url"],
                "chunk_ref": doc["chunk_ref"],
                "page_title": doc["page_title"],
                "crawled_at": doc["crawled_at"]
            } 
            for doc, emb in zip(docs, embeddings)
        ]
        
        # Add to database
        await asyncio.to_thread(table.add, data_to_add)
        
    except Exception as e:
        print(f"  ✗ Failed to process page {page['url']}: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=" * 60)
    print("Documentation Crawler & Ingestion Tool")
    print("=" * 60)
    
    # Show example URLs
    print("\n💡 Example URLs for Zscaler documentation:")
    print("   https://help.zscaler.com/zia")
    print("   https://help.zscaler.com/zpa")
    print("   https://help.zscaler.com/zdx")
    print("   https://help.zscaler.com/ (main documentation site)")
    
    start_url = ""
    while True:
        start_url_input = input("\n🔗 Enter the starting URL (or 'exit'): ").strip()
        
        if start_url_input.lower() == 'exit':
            print("Exiting...")
            break
        
        parsed = urlparse(start_url_input)
        if parsed.scheme and parsed.netloc:
            start_url = start_url_input
            break
        
        print("❌ Invalid URL. Include 'https://' or 'http://'.")
    
    if start_url:
        # Ask about max pages
        max_pages_input = input("\n📄 Maximum pages to crawl (Enter for unlimited): ").strip()
        max_pages = int(max_pages_input) if max_pages_input.isdigit() else None
        
        # For Zscaler, ask if user wants to crawl all product documentation
        additional_paths = []
        if "zscaler.com" in start_url:
            crawl_all = input("\n🔍 Crawl all Zscaler product documentation? (yes/no, default=yes): ").strip().lower()
            if crawl_all != 'no':
                additional_paths = ["/zia", "/zpa", "/zdx", "/zda", "/zcc", "/ztna"]
        
        # Choose crawler mode
        use_playwright = False
        if PLAYWRIGHT_AVAILABLE:
            mode = input("\n🚀 Use JavaScript-enabled browser? (yes/no, default=yes): ").strip().lower()
            use_playwright = mode != 'no'
        else:
            print("\n⚠️  Playwright not available. Using basic crawler.")
            print("   For JavaScript sites, install: pip install playwright && playwright install chromium")
        
        # Database setup
        table = None
        try:
            user_input = input(f"\n⚠️  Clear existing data? (yes/no): ").lower()
            
            if user_input == 'yes':
                print("🗑️  Clearing 'support_docs' table...")
                db.drop_table("support_docs", ignore_missing=True)
                table = db.create_table("support_docs", schema=SupportDoc, mode="overwrite")
                print("✅ Created new table")
            else:
                print("📂 Opening existing table...")
                try:
                    table = db.open_table("support_docs")
                    print("✅ Opened existing table")
                except Exception:
                    print("⚠️  Table not found. Creating new one...")
                    table = db.create_table("support_docs", schema=SupportDoc, mode="overwrite")
        
        except Exception as e:
            print(f"❌ Database error: {e}")
            table = db.create_table("support_docs", schema=SupportDoc, mode="overwrite")
        
        if table is not None:
            # Create state file based on domain
            parsed_url = urlparse(start_url)
            state_file = f"crawl_state_{parsed_url.netloc.replace('.', '_')}.json"
            
            print(f"\n{'=' * 60}")
            print(f"🚀 Starting crawler: {start_url}")
            print(f"📁 State file: {state_file}")
            if additional_paths:
                print(f"📂 Additional paths: {', '.join(additional_paths)}")
            print(f"{'=' * 60}")
            
            async def main():
                if use_playwright:
                    pages_scraped = await crawl_with_playwright(
                        start_url, 
                        max_pages=max_pages, 
                        table=table,
                        state_file=state_file,
                        additional_paths=additional_paths
                    )
                else:
                    print("❌ Basic crawler doesn't support JS sites.")
                    print("   Please install Playwright or use a direct documentation URL.")
                    return
                
                if pages_scraped > 0:
                    print(f"\n{'=' * 60}")
                    print(f"🎉 Done! {pages_scraped} pages have been crawled and saved to the database.")
                    print(f"{'=' * 60}")
                    print(f"\nYour documentation is now searchable.")
                else:
                    print("\n⚠️  No new pages scraped. Check the URL or try increasing the max pages limit.")
            
            asyncio.run(main())