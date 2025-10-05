# KaizenDev Support Bot

This project is a sophisticated, AI-powered Slack bot designed to serve as an intelligent support assistant. It leverages a knowledge base, which it can build by crawling websites, to answer user questions, manage support tickets, and escalate issues to human agents when necessary.

## Features

  * **AI-Powered Responses:** Integrates with OpenAI's GPT models to understand and respond to user queries in natural language.
  * **Knowledge Base Integration:** Uses a LanceDB vector database to store and search a knowledge base of support documentation.
  * **Web Crawler:** Includes a powerful web crawler built with Playwright to automatically populate the knowledge base from websites and documentation portals.
  * **Ticket Management:**
      * `/create-ticket`: Users can create private support tickets.
      * `/close-ticket`: Support agents can archive ticket channels.
  * **Intent Recognition:** Classifies user messages into intents like `question`, `escalation`, `greeting`, etc., to provide appropriate responses.
  * **Conversational Memory:** Remembers the context of conversations within threads or channels to provide more relevant follow-up answers.
  * **Source Citing:** When answering from the knowledge base, the bot cites its sources with links to the original documentation.
  * **DM & Mention Support:** Responds to direct messages and mentions in channels.
  * **App Home:** Provides a welcoming and informative home tab in Slack.

## How It Works

The project is composed of three main components:

1.  **`crawler.py` (The Knowledge Gatherer):**

      * This script crawls a given starting URL (e.g., your documentation site).
      * It uses Playwright to render JavaScript-heavy pages, ensuring all content is captured.
      * It extracts the main content from each page, cleans it, and splits it into smaller, manageable chunks.
      * Each chunk is then converted into a vector embedding using OpenAI's API.
      * These embeddings, along with the original text and source URL, are stored in a LanceDB vector database (`support_db`).

2.  **`ingest.py` (The Manual Librarian):**

      * As an alternative to the crawler, this script can ingest knowledge from local `.txt` files located in a `knowledge` directory.
      * This is useful for adding manually curated documents to the knowledge base.
      * It processes these files, creates embeddings, and adds them to the same LanceDB database.

3.  **`app.py` (The Brains of the Operation):**

      * This is the main Slack bot application.
      * When a user asks a question, the bot first classifies the user's *intent*.
      * If the intent is a `question`, the bot converts the user's query into a vector embedding.
      * It then searches the LanceDB database to find the most relevant text chunks from the knowledge base.
      * The retrieved information is passed to an OpenAI GPT model along with a carefully crafted prompt, instructing it to formulate an answer based *only* on the provided context.
      * If no relevant information is found, the bot uses its general knowledge to answer and suggests escalating to a human.
      * It also handles ticket creation, channel management, and other Slack events.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd slackBot-main
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Playwright browser:**

    ```bash
    playwright install chromium
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add the following:

    ```
    OPENAI_API_KEY="your-openai-api-key"
    SLACK_BOT_TOKEN="your-slack-bot-token"
    SLACK_APP_TOKEN="your-slack-app-token"
    ```

      * `SLACK_BOT_TOKEN`: Your bot's `xoxb-` token.
      * `SLACK_APP_TOKEN`: Your app-level `xapp-` token for Socket Mode.

## Usage

### 1\. Building the Knowledge Base

You have two options for populating the bot's knowledge base.

**Option A: Crawl a Website (Recommended)**

Run the `crawler.py` script to automatically scrape a website.

```bash
python crawler.py
```

The script will prompt you for:

  * **Starting URL:** The URL you want to start crawling from (e.g., `https://help.zscaler.com/zia`).
  * **Maximum pages:** The maximum number of pages to crawl (press Enter for unlimited).
  * **Use JavaScript-enabled browser:** `yes` is recommended for modern websites.
  * **Clear existing data:** `yes` if you want to start with a fresh knowledge base.

**Option B: Ingest Local Files**

1.  Create a folder named `knowledge` in the root directory.
2.  Add your support documents as `.txt` files inside this folder.
3.  Run the `ingest.py` script:
    ```bash
    python ingest.py
    ```
    This script will automatically delete any existing database to avoid duplicates and ingest the content from the `knowledge` folder.

### 2\. Running the Slack Bot

Once the knowledge base (`support_db` directory) is created, you can start the bot:

```bash
python app.py
```

The bot will connect to your Slack workspace via Socket Mode and will be ready to answer questions.

### 3\. Interacting with the Bot in Slack

  * **Ask a question:** Mention the bot in a channel (`@YourBot How do I...`) or send it a direct message.
  * **Create a ticket:** Use the `/create-ticket` command in any channel.
  * **Close a ticket:** Use the `/close-ticket` command inside a ticket channel to archive it.
  * **Get help:** Type `help` in a DM or mention the bot with the word `help`.

## Configuration

  * **Crawling Behavior:** Modify the `should_crawl_url` function and `main_selectors` list in `crawler.py` to customize which links are followed and which HTML elements are targeted for content extraction.
  * **AI Model:** The GPT model can be changed in the `reasoning_engine` function in `app.py` (e.g., from `gpt-5` to another model).
  * **Relevance Threshold:** The `RELEVANCE_THRESHOLD` in `search_knowledge_base` within `app.py` can be adjusted to make the knowledge base search more or less strict.
  * **Conversational Memory:** The number of past exchanges the bot remembers can be configured in the `update_conversation_history` function in `app.py`.