import requests
import re
from bs4 import BeautifulSoup
import csv
import time
import random
import logging
from typing import List, Dict

BASE_URL = "https://www.patpat.lk/en/sri-lanka/vehicle/car"
MAX_PAGES = 322
OUTPUT_FILE = "vehicles_raw.csv"

# Delay settings (ethical scraping)
MIN_DELAY = 2.0
MAX_DELAY = 5.0
REQUEST_TIMEOUT = 30

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def get_soup(url: str) -> BeautifulSoup | None:
    """Fetch a page and return a BeautifulSoup object, or None on error."""
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))  # polite delay
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.content, "lxml")
    except requests.RequestException as e:
        logger.warning(f"Request failed for {url}: {e}")
        return None


def parse_listing_cards(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract basic information from each vehicle listing on a results page.

    During testing in February 2026 we observed that patpat.lk does not expose
    friendly CSS classes for each listing card. Instead, the rendered
    HTML contains a series of anchor (`<a>`) elements whose `href` values
    include the slug `/ad/vehicle/…`.  Each of these anchors wraps all of the
    textual details for a given listing in the following order:

      1. Vehicle title: make, model and year (e.g. ``Toyota Corolla 2008``).
      2. Price or the word ``Negotiable`` (e.g. ``Rs: 3,150,000``).
      3. Posting date and location separated by a vertical bar (e.g.
         ``2026-02-21 | Colombo``).
      4. Mileage in kilometres (e.g. ``167,000 km``).

    By selecting these anchors rather than relying on brittle class names we
    make the scraper more resilient to minor layout changes.  We then parse
    the concatenated text using regular expressions to split the title,
    price, date/location, and mileage into separate fields.

    Returns:
        A list of dictionaries with keys ``Raw_Title``, ``Raw_Price``,
        ``Raw_Mileage``, and ``Raw_Metadata``.
    """
    records: List[Dict[str, str]] = []

    # Use a broad attribute selector to find all vehicle ad links. A
    # typical vehicle listing URL looks like ``/en/ad/vehicle/honda-fit-2015-242846``.
    result_item_selector = "a[href*='/ad/vehicle/']"
    anchors = soup.select(result_item_selector)
    if not anchors:
        logger.warning(
            "No listing anchors found. The site layout may have changed."
        )
        return records

    # Regular expressions to detect a 4‑digit year and an ISO date.
    year_re = re.compile(r"\b(19|20)\d{2}\b")
    date_re = re.compile(r"\d{4}-\d{2}-\d{2}")

    for a in anchors:
        text = a.get_text(separator=" ", strip=True)
        text = " ".join(text.split())  # normalise whitespace
        if not text:
            continue

        parts = text.split()
        # Identify the position of the year in the text. Without a year it's
        # unlikely to be a standard vehicle listing.
        year_idx = None
        for idx, token in enumerate(parts):
            if year_re.fullmatch(token):
                year_idx = idx
                break
        if year_idx is None:
            continue

        title = " ".join(parts[: year_idx + 1])
        price = None
        metadata = None
        mileage = None
        i = year_idx + 1
        # Parse price: either "Negotiable" or an "Rs" amount which may span
        # multiple tokens until a date token is encountered.
        if i < len(parts) and (parts[i].lower().startswith("rs") or parts[i].lower() == "negotiable"):
            price_tokens = [parts[i]]
            i += 1
            while i < len(parts) and not date_re.match(parts[i]):
                price_tokens.append(parts[i])
                i += 1
            price = " ".join(price_tokens)

        # Parse date and location
        if i < len(parts) and date_re.match(parts[i]):
            date_str = parts[i]
            i += 1
            # Skip the separator bar if present
            if i < len(parts) and parts[i] == "|":
                i += 1
            loc_tokens = []
            while i < len(parts) and not parts[i].lower().endswith("km"):
                loc_tokens.append(parts[i])
                i += 1
            location = " ".join(loc_tokens)
            metadata = f"{date_str} | {location}" if location else date_str
            # Parse mileage (two tokens: number and 'km')
            if i + 1 <= len(parts):
                mileage_tokens = parts[i : i + 2]
                mileage = " ".join(mileage_tokens).strip()

        # Determine the absolute URL for this listing.  Some anchors may
        # include a relative path (starting with '/') so we join it against
        # the domain of BASE_URL.  If the href is already absolute we use it
        # directly.
        href = a.get("href", "")
        full_url = href
        # Only attempt to construct absolute URLs when the href is relative.
        if href and href.startswith("/"):
            try:
                from urllib.parse import urljoin

                full_url = urljoin(BASE_URL, href)
            except Exception:
                full_url = href

        record = {
            "Raw_Title": title or None,
            "Raw_Price": price or None,
            "Raw_Mileage": mileage or None,
            "Raw_Metadata": metadata or None,
            "Listing_URL": full_url,
        }
        # Append all listings regardless of price; filtering is done later.  Each
        # record includes the Listing_URL so we can fetch additional
        # attributes from the detail page when needed.
        if any(record.values()):
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Detail Page Parsing
#
# Each vehicle listing has its own page (e.g. ``/en/ad/vehicle/honda-civic-2017-241953``) with
# additional details such as mileage, engine capacity, transmission, etc.  The
# ``get_vehicle_details`` function fetches the detail page and parses out
# common information from the body text.  These details are merged into the
# final record when ``scrape_patpat`` processes the search results.

DETAIL_LABELS = [
    "Mileage",
    "Engine/Motor Capacity",
    "Transmission",
    "Manufacturer",
    "Model Year",
    "Condition",
    "Model",
    "Fuel Type",
    "Colour",
    "Vehicle Type",
    "Power",
    "Register Year",
]


def get_vehicle_details(url: str) -> Dict[str, str]:
    """
    Fetch a vehicle detail page and extract common attributes.

    Args:
        url: Absolute URL of the vehicle advertisement.

    Returns:
        A dictionary mapping attribute names (e.g. ``Mileage``, ``Engine/Motor Capacity``)
        to their corresponding values.  If a page cannot be fetched due to a
        network error or unexpected structure, an empty dict is returned.
    """
    details: Dict[str, str] = {}
    soup = get_soup(url)
    if soup is None:
        return details

    # Extract text from the page.  We normalise whitespace and split into
    # individual lines to facilitate sequential label/value parsing.  Many
    # labels and values appear on consecutive lines in the order defined in
    # ``DETAIL_LABELS``.
    full_text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
    # Build a lookup from label to value by scanning the lines sequentially.
    for i, line in enumerate(lines[:-1]):
        for label in DETAIL_LABELS:
            # Match the exact label (case insensitive) at the start of the line
            if line.lower() == label.lower():
                value = lines[i + 1]
                details[label] = value
                break

    return details


def scrape_patpat(
    max_pages: int = MAX_PAGES,
    fetch_details: bool = True,
) -> List[Dict[str, str]]:
    """
    Paginate through patpat.lk and collect listing data.

    If ``fetch_details`` is True, additional attributes are fetched from each
    individual vehicle page for listings that include a concrete price (i.e.
    not ``Negotiable``).  Listings without a price are skipped entirely.

    Args:
        max_pages: Number of search result pages to scrape.
        fetch_details: Whether to follow each listing URL and extract
            detailed attributes.  Turning this off speeds up crawling but
            yields only the basic fields from the search results.

    Returns:
        A list of dictionaries representing vehicle listings.  Each record
        contains the raw fields (title, price, mileage, metadata) and, if
        ``fetch_details`` is enabled, any additional attributes present on
        the detail page (e.g. ``Mileage``, ``Transmission``, etc.).
    """
    all_records: List[Dict[str, str]] = []

    for page in range(1, max_pages + 1):
        # Construct the URL for each page.  The first page uses the base URL
        # without a query parameter; subsequent pages append ``?page=N``.
        if page == 1:
            url = BASE_URL
        else:
            url = f"{BASE_URL}?page={page}"

        logger.info(f"Scraping page {page}/{max_pages}: {url}")
        soup = get_soup(url)
        if soup is None:
            logger.warning(f"Skipping page {page} due to fetch error.")
            continue

        page_records = parse_listing_cards(soup)
        logger.info(f"Found {len(page_records)} records on page {page}")

        for record in page_records:
            price_str = (record.get("Raw_Price") or "").strip().lower()
            # Skip listings without a price or with the word "negotiable".  A
            # non-empty price that does not begin with "negotiable" indicates
            # a numeric price is available (e.g. ``Rs: 5,000,000``).
            if not price_str or price_str.startswith("negotiable"):
                continue

            # Fetch additional details from the listing page if requested.
            if fetch_details:
                listing_url = record.get("Listing_URL")
                if listing_url:
                    details = get_vehicle_details(listing_url)
                    # Merge the detail fields into the record.  Note that
                    # detail keys may overlap with raw field names; in such
                    # cases the detail value overwrites the raw value.
                    record.update(details)

            # Remove the Listing_URL field from the final record; it isn't
            # required for machine learning but may be useful for debugging.  If
            # you wish to retain the URL, comment out the next line.
            record.pop("Listing_URL", None)
            all_records.append(record)

    logger.info(f"Total records scraped (after filtering): {len(all_records)}")
    return all_records


def save_to_csv(records: List[Dict[str, str]], filepath: str = OUTPUT_FILE) -> None:
    """
    Save a list of dictionaries to a CSV file.

    The CSV header is automatically derived from the union of all keys present
    in ``records``.  Records missing a field will have a blank cell in the
    output.  If no records are provided the function logs a warning and
    does not create a file.

    Args:
        records: List of records returned from ``scrape_patpat``.
        filepath: Destination file path for the CSV.
    """
    if not records:
        logger.warning("No records to save. CSV will not be created.")
        return

    # Determine the complete set of field names across all records.  Use a
    # deterministic order by sorting the fieldnames alphabetically, but keep
    # the four core raw fields at the front for readability.
    all_keys = set()
    for rec in records:
        all_keys.update(rec.keys())
    # Remove Listing_URL if it somehow remains.
    all_keys.discard("Listing_URL")
    core_fields = ["Raw_Title", "Raw_Price", "Raw_Mileage", "Raw_Metadata"]
    other_fields = sorted(k for k in all_keys if k not in core_fields)
    fieldnames = core_fields + other_fields

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Saved {len(records)} records to {filepath}")


if __name__ == "__main__":
    logger.info("Starting patpat.lk scraping...")
    data = scrape_patpat(MAX_PAGES)
    save_to_csv(data, OUTPUT_FILE)
    logger.info("Scraping finished.")