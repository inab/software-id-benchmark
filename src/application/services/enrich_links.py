import requests
import urllib.parse
import os
import logging
import json
import re
from readability import Document
from bs4 import BeautifulSoup
#from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

# -------------------------------
# Configuration & Logging
# -------------------------------

# Github API
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

logging.basicConfig(level=logging.INFO)

# Github metadata api
GITHUB_METADATA_URL = "https://observatory.openebench.bsc.es/github-metadata-api/metadata/user"
GITHUB_CONTENT_URL = "https://observatory.openebench.bsc.es/github-metadata-api/metadata/content/user"

# Gitlab API
GITLAB_TOKEN = os.environ.get("GITLAB_TOKEN")

# -------------------------------
# GitHub API Helpers
# -------------------------------

def request_github_metadata(owner, repo_name):
    data = {
        'owner': owner,
        'repo': repo_name,
        'userToken': GITHUB_TOKEN,
        'prepare': False
    }
    try:
        response = requests.post(GITHUB_METADATA_URL, data=data)
        response.raise_for_status()
        return response.json().get('data')
    except Exception as e:
        logging.warning(f"Metadata fetch failed for {owner}/{repo_name}: {e}")
        return None

def request_github_content(owner, repo_name, file_path):
    data = {
        'owner': owner,
        'repo': repo_name,
        'path': file_path,
        'userToken': GITHUB_TOKEN
    }
    try:
        response = requests.post(GITHUB_CONTENT_URL, data=data)
        response.raise_for_status()
        return response.json().get('content')
    except Exception as e:
        logging.warning(f"README fetch failed for {owner}/{repo_name}: {e}")
        return None

def request_github_readme(owner, repo_name):
    try:
        # Step 1: List all files in the repo root
        contents_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}/contents/"
        response = requests.get(contents_url, headers=GITHUB_API_HEADERS)
        response.raise_for_status()
        files = response.json()

        # Step 2: Look for README file, case-insensitive
        readme_file = next(
            (f for f in files if f["type"] == "file" and f["name"].lower().startswith("readme")),
            None
        )

        if not readme_file:
            logging.info(f"No README found for {owner}/{repo_name}")
            return None

        # Step 3: Fetch the content of the README file
        #  extract path
        readme_path = readme_file["path"]
        content = request_github_content(owner, repo_name, readme_path)
        if not content:
            logging.info(f"README content not found for {owner}/{repo_name}")
            return None
        
        return content
    
    except Exception as e:
        logging.warning(f"README fetch failed for {owner}/{repo_name}: {e}")
        return None
    
# -------------------------------
# GitHub API Helpers
# -------------------------------


def parse_gitlab_repo_url(repo_url: str) -> str:
    """
    Extracts and URL-encodes the namespace/repo string from a GitLab repository URL.

    For example, if repo_url is 'https://gitlab.com/mygroup/myrepo',
    this returns 'mygroup%2Fmyrepo'.
    """
    # A simple regex to capture the namespace and repo name from a GitLab URL.
    pattern = r"https?://gitlab\.com/([^/]+/[^/]+)"
    match = re.search(pattern, repo_url)
    if not match:
        raise ValueError("Invalid GitLab repository URL.")
    namespace_repo = match.group(1)
    return urllib.parse.quote(namespace_repo, safe="")

def get_gitlab_repo_metadata(repo_url: str) -> dict:
    """
    Given a GitLab repository URL, returns the repository metadata as a dictionary.
    
    Optionally accepts a GitLab Personal Access Token for private repositories.
    """
    encoded_project = parse_gitlab_repo_url(repo_url)
    api_url = f"https://gitlab.com/api/v4/projects/{encoded_project}"
    
    headers = {}
    if GITLAB_TOKEN:
        headers["PRIVATE-TOKEN"] = GITLAB_TOKEN

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to get metadata. Status code: {response.status_code}, Response: {response.text}")
    
    return response.json()

def get_gitlab_repo_readme(readme_url: str, repo_url: str) -> str:
    """
    Attempts to retrieve the README content from a GitLab repository.

    Returns the content of the first found README file. Raises an error if none are found.
    """
    logging.info(f"Fetching README from GitLab: {readme_url}")
    try:
        readme_fields = readme_url.split("/")
        default_branch = readme_fields[-2]
        file_name = readme_fields[-1]

        encoded_project = parse_gitlab_repo_url(repo_url)

        headers = {}
        if GITLAB_TOKEN:
            headers["PRIVATE-TOKEN"] = GITLAB_TOKEN

        api_url = (
            f"https://gitlab.com/api/v4/projects/{encoded_project}/repository/files/{file_name}/raw"
        )
        params = {"ref": default_branch}

        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == 200:
            return response.text  # Return first successful content
        else:
            logging.warning(f"README not found at {api_url}. Trying other common filenames.")
            return None
    
    except Exception as e:
        logging.warning(f"Error fetching README from GitLab: {e}")
        return None


# -------------------------------
# Web Scraping & Parsing
# -------------------------------
async def get_link_content(link):
    decoded_link = urllib.parse.unquote(link)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.90 Safari/537.36"
    }

    if "galaxy.bi.uni-freiburg.de/tool_runner" in decoded_link:
        decoded_link = decoded_link.replace("galaxy.bi.uni-freiburg.de/tool_runner", "usegalaxy.eu/root")

    result = await extract_with_playwright(decoded_link)
    return result
    

def normalize_linebreaks(text: str) -> str:
    # Replace any escaped "\n" with real newlines
    text = text.replace('\\n', '\n')

    # Replace any Windows-style line endings with Unix-style
    text = text.replace('\r\n', '\n')

    # Optionally collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def extract_main_text_from_html(html: str) -> str:
    doc = Document(html)
    summary_html = doc.summary()
    soup = BeautifulSoup(summary_html, 'html.parser')

    # Convert links to [text](url)
    for a in soup.find_all('a', href=True):
        text = a.get_text(strip=True)
        href = a['href']
        if text:
            a.replace_with(f"[{text}]({href})")
        else:
            a.replace_with(f"<{href}>")

    # Convert <strong>/<b> to **bold**
    for tag in soup.find_all(['strong', 'b']):
        text = tag.get_text(strip=True)
        tag.replace_with(f"**{text}**")

    # Convert <em>/<i> to _italic_
    for tag in soup.find_all(['em', 'i']):
        text = tag.get_text(strip=True)
        tag.replace_with(f"_{text}_")

    # Convert headers to Markdown (#, ##, etc.)
    for i in range(1, 7):
        for header in soup.find_all(f'h{i}'):
            text = header.get_text(strip=True)
            header.replace_with(f"\n{'#' * i} {text}\n")

    # Flatten lists as bullets
    for li in soup.find_all('li'):
        text = li.get_text(strip=True)
        li.replace_with(f"* {text}")

    # Remove scripts/styles/irrelevant
    for tag in soup(['script', 'style', 'footer', 'nav']):
        tag.decompose()

    # Clean paragraphs and line breaks
    text = soup.get_text(separator='\n', strip=True)

    # Normalize line breaks
    text = normalize_linebreaks(text)

    with open(f"content_clean.html", "w", encoding="utf-8") as f:
        f.write(text)
    
    return text


def extract_sourceforge_project_info(html: str) -> dict:
    logging.info("Extracting SourceForge project info")

    result = {
        "description": None,
        "sections": []  # List of dicts with text and hrefs from each psp-section
    }

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Extract project description
        desc_tag = soup.find("p", class_="description")
        if desc_tag:
            result["description"] = desc_tag.get_text(separator="\n", strip=True)

        # Extract sections
        # Note: SourceForge uses both "div" and "section" with class "psp-section"
        for section in soup.find_all("div", class_="psp-section"):
            # Extract text and links from the outer section
            section_text = section.get_text(separator="\n", strip=True)
            links = [a["href"] for a in section.find_all("a", href=True)]

            # Check for any internal divs and extract their text and links too
            inner_divs = section.find_all("div", recursive=False)
            for inner in inner_divs:
                section_text += "\n" + inner.get_text(separator="\n", strip=True)
                links.extend([a["href"] for a in inner.find_all("a", href=True)])

            # Remove duplicates and clean links
            links = list(set(links))

            result["sections"].append({
                "text": section_text.strip(),
                "hrefs": links
            })

        for section in soup.find_all("section", class_="psp-section"):
            # Extract text and links from the outer section
            section_text = section.get_text(separator="\n", strip=True)
            links = [a["href"] for a in section.find_all("a", href=True)]

            # Check for any internal divs and extract their text and links too
            inner_divs = section.find_all("div", recursive=False)
            for inner in inner_divs:
                section_text += "\n" + inner.get_text(separator="\n", strip=True)
                links.extend([a["href"] for a in inner.find_all("a", href=True)])

            # Remove duplicates and clean links
            links = list(set(links))

            result["sections"].append({
                "text": section_text.strip(),
                "hrefs": links
            })

    except Exception as e:
        logging.warning(f"Error parsing SourceForge HTML: {e}")

    return result


def get_pypi_project_info(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    logging.info(f"Fetching PyPI metadata for {package_name}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        info = data.get("info", {})
        # remove null fields 
        for key in list(info.keys()):
            if info[key] is None:
                del info[key]

        info['releases'] = [] 
        for key in data.get("releases", {}):
            info['releases'].append(key)
            
        return info

    except Exception as e:
        logging.warning(f"Error fetching PyPI data: {e}")
        return None
    
def get_bitbucket_metadata(user, repo):
    """Get metadata from a Bitbucket repository."""
    try:
        logging.info(f"Extracting metadata from Bitbucket repository {user}/{repo}")
        api_url = f"https://api.bitbucket.org/2.0/repositories/{user}/{repo}"
        response = requests.get(api_url, timeout=10)
        if response.status_code != 200:
            logging.warning(f"Failed to fetch metadata for {user}/{repo}: {response.status_code}")
            return {"error": f"Failed to fetch metadata: {response.status_code}"}
        data = response.json()
        logging.info(f"Metadata fetched successfully for {user}/{repo}")

        return data 
    
    except Exception as e:
        return {"error": str(e)}

def get_bitbucket_readme(user, repo, metadata):
    """Try to fetch README content from a Bitbucket repository."""
    try:
        main_branch = metadata.get("main_branch") or "master"

        # Try common readme filenames
        readme_candidates = ["README.md", "README.rst", "README.txt", "readme.md", "readme.rst", "readme.txt"]

        for filename in readme_candidates:
            raw_url = f"https://bitbucket.org/{user}/{repo}/raw/{main_branch}/{filename}"
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200 and response.text.strip():
                logging.info(f"README found at {raw_url}: {response.text}")
                return response.text
            
        logging.warning(f"No README found for {user}/{repo} in common locations.")

        return None  # No readme found
    except Exception as e:
        return None

async def extract_with_playwright(url: str) -> str:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, args=[
                "--proxy-bypass-list=<-loopback>",
                "--dns-prefetch-disable"
            ])
            context = await browser.new_context()
            page = await context.new_page()
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."
            })

            response = await page.goto(url, wait_until="domcontentloaded")
            if response and response.status != 200:
                logging.warning(f"Non-200 status: {response.status} for {url}")
                await browser.close()
                return None

            content = await page.content()
            # save content in a file
            with open(f"content.html", "w", encoding="utf-8") as f:
                f.write(content)

            await browser.close()

            return content

    except Exception as e:
        logging.warning(f"Playwright failed for {url}: {e}")
        return None
    
# -------------------------------
# Repository/Webpage Enrichment
# -------------------------------

def enrich_repo(url):
    repo = { 'url': url ,'metadata': None, 'readme_content': None }
    try:
        parts = url.split('/')
        owner, repo_name = parts[3], parts[4]
        logging.info(f"Fetching GitHub metadata for {owner}/{repo_name} with token {GITHUB_TOKEN[:4]}...")
        repo['repo_metadata'] = request_github_metadata(owner, repo_name)
        repo['readme_content'] = request_github_readme(owner, repo_name)
    except Exception as e:
        logging.error(f"Invalid GitHub URL: {url} -> {e}")

    return repo

def get_redirect(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.url
    except Exception as e:
        return False

    
async def enrich_link(link):
    new_link = {'url': link}
    link = get_redirect(link)

    if link:
        processed = False
        if "github.com" in link:
            try:
                parts = link.split('/')
                owner, repo_name = parts[3], parts[4]
                new_link['repo_metadata'] = request_github_metadata(owner, repo_name)
                new_link['readme_content'] = request_github_readme(owner, repo_name)
                processed = True
            except Exception as e:
                logging.warning(f"Error processing GitHub link {link}: {e}")

        elif "gitlab.com" in link:
            pattern = r"https?://gitlab\.com/([^/]+/[^/]+)"
            match = re.search(pattern, link)
            metadata = get_gitlab_repo_metadata(link)
            new_link['repo_metadata'] = metadata

            if metadata:
                readme_url = metadata.get('readme_url')
                if readme_url:
                    new_link['readme_content'] = get_gitlab_repo_readme(readme_url, link)
                    processed = True


        elif "pypi.org/project/" in link:
            # Extract package name from the URL
            # Example: https://pypi.org/project/package_name/foo
            
            package_name = link.split("pypi.org/project/")[1]
            package_name = package_name.split("/")[0]
            metadata = get_pypi_project_info(package_name)
            new_link['project_metadata'] = metadata
            processed = True
        
        sourceforge_alternatives = ["sourceforge.net/projects/", 'sf.net/p/', 'sourceforge.net/p/']
        if any(alt in link for alt in sourceforge_alternatives):
            try:
                # Extract project info from SourceForge
                content = await get_link_content(link)
                # write content to file 
                    
                project_info = extract_sourceforge_project_info(content)
                new_link['project_metadata'] = project_info 
                processed = True
            except Exception as e:
                logging.warning(f"Error processing SourceForge link {link}: {e}")


        elif "bitbucket.org" in link:
            try:
                match = re.match(r"https?://bitbucket.org/([^/]+)/([^/]+)", link)
                user, repo = match.groups() if match else (None, None)
                metadata = get_bitbucket_metadata(user, repo)
                new_link['repo_metadata'] = metadata
                if metadata and 'main_branch' in metadata:
                    new_link['readme_content'] = get_bitbucket_readme(user, repo, metadata)
                processed = True
       
            except Exception as e:
                logging.warning(f"Error processing Bitbucket link {link}: {e}")

        elif "git.bioconductor" in link:
            new_link['repo_metadata'] = {'url': link}
            processed = True

        if not processed:
            logging.info(f"Extracting generic content from {link}")
            content = await get_link_content(link)
            if content:
                text = extract_main_text_from_html(content)
                new_link['content'] = text 

    #logging.info(f"Enriched link:")
    #logging.info(json.dumps(new_link, indent=2))

    return new_link


