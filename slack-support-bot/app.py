import os
import asyncio
import logging
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from dotenv import load_dotenv

# Import AI and DB libraries
import lancedb
import openai
from lancedb.pydantic import LanceModel, Vector

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- INITIALIZATION ---
app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))
openai.api_key = os.environ.get("OPENAI_API_KEY")
db = lancedb.connect("./support_db")

# --- DATA MODEL ---
class SupportDoc(LanceModel):
    text: str
    vector: Vector(1536)
    source_url: str

try:
    table = db.open_table("support_docs")
    logger.info("LanceDB table 'support_docs' opened successfully.")
except Exception:
    table = db.create_table("support_docs", schema=SupportDoc)
    logger.info("LanceDB table 'support_docs' created successfully.")

# --- THE BOT'S "BRAIN": LONG-TERM CONVERSATIONAL MEMORY ---
conversation_memory = {}

def get_conversation_history(key: str):
    if key not in conversation_memory:
        conversation_memory[key] = []
    return conversation_memory[key]

def update_conversation_history(key: str, role: str, content: str):
    if key not in conversation_memory:
        conversation_memory[key] = []
    conversation_memory[key].append({"role": role, "content": content})

# --- CORE LOGIC HELPER ---
async def _process_mention(event, say, context, logger, query: str, channel_id: str, thread_ts: str = None):
    """Shared logic for handling both new mentions and edited mentions."""
    try:
        # Determine the correct key for memory and the correct parameters for replying.
        if thread_ts:
            memory_key = thread_ts
            reply_params = {"channel": channel_id, "thread_ts": memory_key}
        else:
            memory_key = channel_id
            reply_params = {"channel": channel_id}

        logger.info(f"--- PROCESSING MENTION ---")
        logger.info(f"User: <@{event['user']}> | Memory Key: {memory_key} | Query: '{query}'")

        # 1. Get conversation history
        history = get_conversation_history(memory_key)
        logger.info(f"Retrieved {len(history)} messages from memory.")

        # 2. Classify Intent
        intent = await classify_intent(query)
        logger.info(f"Classified Intent: '{intent}'")

        # 3. Handle based on intent
        if intent == "question":
            await say(text=f"Sure <@{event['user']}>, let me look that up for you...", **reply_params)
            answer = await reasoning_engine(query, history)
            logger.info(f"Generated Answer: '{answer[:150]}...'")
            await say(text=answer, **reply_params)
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", answer)

        elif intent == "escalation":
            logger.info("Escalation request detected.")
            await say(text=f"No problem <@{event['user']}>, I'm escalating this to a human agent for you. They will be with you shortly.", **reply_params)
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", "Escalated to human agent.")

        elif intent == "follow_up":
            logger.info("Follow-up message detected.")
            await say(text=f"You're welcome, <@{event['user']}>! Let me know if you need anything else.", **reply_params)
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", "Acknowledged follow-up.")

        else: # 'other'
            logger.warning(f"Unhandled intent '{intent}' for query: '{query}'")
            await say(text=f"I'm not sure how to respond to that, <@{event['user']}>. Can you try rephrasing your question?", **reply_params)
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", "Asked for clarification.")

        logger.info(f"--- END MENTION ---\n")

    except Exception as e:
        logger.error(f"Unhandled error in _process_mention: {e}", exc_info=True)
        await say(text="Sorry, I encountered an unexpected error. Please try again later.", channel=channel_id)


# --- SLACK EVENT HANDLERS ---
@app.event("app_home_opened")
async def update_home_tab(client, event, logger):
    try:
        await client.views_publish(user_id=event["user"], view={
            "type": "home", "blocks": [{
                "type": "section", "text": {"type": "mrkdwn", "text": "*Welcome to the KaizenDev Support Bot!* :robot_face:\n\nI'm smarter now! I remember our conversations, provide detailed answers with source citations, and respect API limits. Just mention me. I can also understand edited messages!"}
            }]
        })
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")

@app.event("app_mention")
async def handle_app_mention(event, say, context, logger):
    """Handles new mentions."""
    query = event["text"].split(f"<@{context.bot_user_id}>")[-1].strip()
    channel_id = event["channel"]
    thread_ts = event.get("thread_ts")
    await _process_mention(event, say, context, logger, query, channel_id, thread_ts)

@app.event("message")
async def handle_message_event(event, say, context, logger):
    """Handles message subtypes, like 'message_changed'."""
    # Check if the event is a message edit
    if event.get("subtype") == "message_changed":
        # Check if the bot is mentioned in the NEW version of the message
        new_text = event["message"]["text"]
        if f"<@{context.bot_user_id}>" in new_text:
            logger.info("Detected bot mention in an edited message.")
            query = new_text.split(f"<@{context.bot_user_id}>")[-1].strip()
            channel_id = event["channel"]
            thread_ts = event["message"].get("thread_ts")
            await _process_mention(event, say, context, logger, query, channel_id, thread_ts)


# --- CORE LOGIC FUNCTIONS (THE BRAIN) ---
async def reasoning_engine(query: str, history: list):
    search_results = await search_knowledge_base(query)
    history_str = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history])

    prompt = f"""
    You are a world-class expert support assistant for KaizenDev. Your goal is to provide a detailed, yet concise, answer within a 500-token limit.

    **Instructions:**
    1.  **Analyze:** Review the user's request and conversation history.
    2.  **Synthesize:** Use the provided 'Context' from the knowledge base and your own expertise.
    3.  **Answer:** Provide a clear, step-by-step, and detailed answer. Be comprehensive but avoid fluff to stay within the token limit.
    4.  **Cite Sources:** At the end of your answer, you MUST include a "References" section with the URLs provided in the context. Format it like:
        > References:
        > - <URL_1>
        > - <URL_2>
    5.  **Be Proactive:** End with a helpful follow-up question if the user's issue might not be fully resolved.

    **Conversation History:**
    ---
    {history_str}
    ---

    **Knowledge Base Context (with Sources):**
    ---
    {search_results}
    ---

    **User's Current Request:**
    ---
    {query}
    ---

    **Your Detailed Response with References:**
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a world-class expert support assistant. Follow instructions strictly, especially the token limit and citation format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        answer = answer.replace('**', '*')
        return answer
    except Exception as e:
        logger.error(f"Error in reasoning engine: {e}")
        return "My brain had a short circuit. Can you try asking that again?"

async def search_knowledge_base(query: str):
    try:
        response = openai.embeddings.create(input=query, model="text-embedding-ada-002")
        query_vector = response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return "No information found due to an error."

    try:
        results = table.search(query_vector).limit(3).to_pandas()
        if results.empty:
            return "No relevant information found in the knowledge base."
        
        context_list = []
        for index, row in results.iterrows():
            context_list.append(f"Source: {row['source_url']}\nContent: {row['text']}")
        return "\n\n---\n\n".join(context_list)
    except Exception as e:
        logger.error(f"Error searching database: {e}")
        return "No information found due to an error."

async def classify_intent(text: str):
    """
    Classifies the user's intent using a robust few-shot prompt.
    """
    prompt = f"""
    You are an expert at classifying user intent for a support bot.
    Analyze the user's message and classify its intent into ONE of these categories: 'question', 'escalation', 'follow_up', or 'other'.

    Here are some examples to guide you:

    Message: "how do I reset my password?"
    Intent: question

    Message: "I need to speak to a manager now!"
    Intent: escalation

    Message: "thanks, that worked"
    Intent: follow_up

    Message: "got it, thanks"
    Intent: follow_up

    Message: "what's the weather like?"
    Intent: other

    ---
    Now, classify the following message:

    Message: "{text}"

    Intent:
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at classifying user intent. Follow the examples provided strictly and respond with only the intent word."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        intent = response.choices[0].message.content.strip().lower()
        valid_intents = ["question", "escalation", "follow_up", "other"]
        return intent if intent in valid_intents else "other"
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        return "other"

# --- MAIN EXECUTION ---
async def main():
    logger.info("Starting KaizenDev Support Bot with Source Citations and Edit Support...")
    handler = AsyncSocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    await handler.start_async()

if __name__ == "__main__":
    asyncio.run(main())