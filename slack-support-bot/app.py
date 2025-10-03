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
    try:
        if thread_ts:
            memory_key = thread_ts
            reply_params = {"channel": channel_id, "thread_ts": memory_key}
        else:
            memory_key = channel_id
            reply_params = {"channel": channel_id}

        user_id = event.get('user') or event.get('message', {}).get('user')
        logger.info(f"--- PROCESSING MENTION ---")
        logger.info(f"User: <@{user_id}> | Memory Key: {memory_key} | Query: '{query}'")

        history = get_conversation_history(memory_key)
        intent = await classify_intent(query)
        logger.info(f"Classified Intent: '{intent}'")

        if intent == "question":
            await say(text=f"Sure <@{user_id}>, let me look that up for you...", **reply_params)
            answer = await reasoning_engine(query, history, user_id)
            await say(text=answer, **reply_params)
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", answer)
        elif intent == "escalation":
            await say(text=f"No problem <@{user_id}>, I'm escalating this to a human agent for you.", **reply_params)
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", "Escalated to human agent.")
        elif intent == "follow_up":
            await say(text=f"You're welcome, <@{user_id}>! Let me know if you need anything else.", **reply_params)
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", "Acknowledged follow-up.")
        else:
            await say(text=f"I'm not sure how to respond to that, <@{user_id}>. Can you try rephrasing your question?", **reply_params)
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
            "type": "home", "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "Welcome! I'm your support assistant. Mention me with any question."}}]
        })
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")

@app.event("app_mention")
async def handle_app_mention(event, say, context, logger):
    query = event["text"].split(f"<@{context.bot_user_id}>")[-1].strip()
    await _process_mention(event, say, context, logger, query, event["channel"], event.get("thread_ts"))

@app.event("message")
async def handle_message_event(event, say, context, logger):
    if event.get("subtype") == "message_changed":
        new_text = event["message"]["text"]
        if f"<@{context.bot_user_id}>" in new_text:
            query = new_text.split(f"<@{context.bot_user_id}>")[-1].strip()
            await _process_mention(event, say, context, logger, query, event["channel"], event["message"].get("thread_ts"))

# --- CORE LOGIC FUNCTIONS (THE BRAIN) ---
async def reasoning_engine(query: str, history: list, user_id: str):
    search_results = await search_knowledge_base(query)
    history_str = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history])

    if "No relevant information found" in search_results:
        # General knowledge prompt, without asking for URLs
        prompt = f"""
        You are a world-class expert support assistant for KaizenDev.
        The user <@{user_id}> has asked a question for which there is no specific information in your internal knowledge base.
        Provide a helpful and comprehensive answer based on your own expertise. Do not cite any sources.

        **Conversation History:**
        ---
        {history_str}
        ---

        **User's Current Request:**
        ---
        {query}
        ---

        **Your General Response:**
        """
    else:
        # Knowledge base prompt, which requires citing the provided source
        prompt = f"""
        You are a world-class expert support assistant for KaizenDev.
        Your goal is to provide a detailed answer based ONLY on the provided context.

        **Instructions:**
        1.  Synthesize the 'Knowledge Base Context' to formulate your answer. Do not use outside knowledge.
        2.  At the end of your answer, you MUST include a "References" section with the URLs provided in the context.

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

        **Your Detailed Response based ONLY on the Provided Context:**
        """
    try:
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": "You are a helpful expert assistant."}, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )
        answer = response.choices[0].message.content.strip()
        return answer.replace('**', '*')
    except Exception as e:
        logger.error(f"Error in reasoning engine: {e}")
        return "My brain had a short circuit. Can you try asking that again?"

async def search_knowledge_base(query: str):
    RELEVANCE_THRESHOLD = 0.6
    try:
        response = await asyncio.to_thread(openai.embeddings.create, input=query, model="text-embedding-ada-002")
        query_vector = response.data[0].embedding
        
        results_df = await asyncio.to_thread(table.search(query_vector).limit(3).to_pandas)
        relevant_results = results_df[results_df['_distance'] <= RELEVANCE_THRESHOLD]

        if relevant_results.empty:
            return "No relevant information found in the knowledge base."
        
        context_list = [f"Source: {row['source_url']}\nContent: {row['text']}" for _, row in relevant_results.iterrows()]
        return "\n\n---\n\n".join(context_list)
    except Exception as e:
        logger.error(f"Error during knowledge base search: {e}")
        return "No information found due to an error."

async def classify_intent(text: str):
    prompt = f"""
    You are an expert at classifying user intent. Classify the message into ONE of these categories: 'question', 'escalation', 'follow_up', or 'other'.
    Your response must be a single word.

    Examples:
    Message: "how do I reset my password?" -> Intent: question
    Message: "I have ZCC installed and my application is slowing down. what might be the cause?" -> Intent: question
    Message: "I need to speak to a manager now!" -> Intent: escalation
    Message: "thanks, that worked" -> Intent: follow_up
    Message: "what's the weather like?" -> Intent: other
    ---
    Classify the following message:
    Message: "{text}"
    Intent:
    """
    try:
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at classifying user intent. Respond with only a single word: question, escalation, follow_up, or other."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        intent = response.choices[0].message.content.strip().lower()
        intent = ''.join(c for c in intent if c.isalpha())
        
        valid_intents = ["question", "escalation", "follow_up", "other"]
        return intent if intent in valid_intents else "other"
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        return "other"

# --- MAIN EXECUTION ---
async def main():
    logger.info("Starting KaizenDev Support Bot...")
    handler = AsyncSocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    await handler.start_async()

if __name__ == "__main__":
    asyncio.run(main())