import os
import asyncio
import logging
import re
import json
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
    chunk_ref: str = ""
    page_title: str = ""
    crawled_at: str = ""

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
    
    # Keep only the last 10 exchanges to manage token usage
    if len(conversation_memory[key]) > 20:
        conversation_memory[key] = conversation_memory[key][-20:]

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
        
        # Enhanced logging for query processing
        print("\n" + "="*80)
        print(f"🔍 PROCESSING NEW QUERY")
        print("="*80)
        print(f"User: <@{user_id}>")
        print(f"Channel: {channel_id}")
        print(f"Thread: {thread_ts if thread_ts else 'N/A (Main Channel)'}")
        print(f"Query: '{query}'")
        print("-"*80)
        
        history = get_conversation_history(memory_key)
        intent = await classify_intent(query)
        
        print(f"Classified Intent: '{intent}'")
        print("-"*80)

        if intent == "question":
            await say(text=f"Sure <@{user_id}>, let me look that up for you...", **reply_params)
            answer = await reasoning_engine(query, history, user_id)
            await say(**answer, **reply_params)
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", answer.get("text", ""))
        elif intent == "escalation":
            await say(
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"No problem <@{user_id}>, I'm escalating this to a human agent for you."
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": "A human agent will respond shortly. Please provide any additional details that might help."
                            }
                        ]
                    }
                ],
                **reply_params
            )
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", "Escalated to human agent.")
        elif intent == "follow_up":
            await say(
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"You're welcome, <@{user_id}>! :smile:"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": "Let me know if you need anything else!"
                            }
                        ]
                    }
                ],
                **reply_params
            )
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", "Acknowledged follow-up.")
        elif intent == "greeting":
            await say(
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Hello <@{user_id}>! :wave:"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "How can I help you today? I can assist with:\n• Technical questions\n• Troubleshooting\n• Product information\n• General support"
                        }
                    }
                ],
                **reply_params
            )
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", "Greeting acknowledged.")
        else:
            await say(
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"I'm not sure how to respond to that, <@{user_id}>. :thinking_face:"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Could you try rephrasing your question? Here are some examples:\n• \"How do I reset my password?\"\n• \"What are the system requirements?\"\n• \"I'm having trouble with the installation\""
                        }
                    }
                ],
                **reply_params
            )
            update_conversation_history(memory_key, "user", query)
            update_conversation_history(memory_key, "assistant", "Asked for clarification.")

        print(f"✅ Query processed and response sent")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Unhandled error in _process_mention: {e}", exc_info=True)
        await say(
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Sorry, I encountered an unexpected error. :x:"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "Please try again later or escalate to a human agent."
                        }
                    ]
                }
            ],
            channel=channel_id
        )

# --- SLACK EVENT HANDLERS ---
@app.event("app_home_opened")
async def update_home_tab(client, event, logger):
    try:
        await client.views_publish(user_id=event["user"], view={
            "type": "home", 
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "Welcome to KaizenDev Support! 🚀"}},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn", "text": "I'm your AI support assistant. Mention me in any channel with your questions, and I'll help you find answers from our knowledge base or provide general assistance."}},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn", "text": "*How to use me:*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "• Simply mention me with your question: `@YourBot How do I reset my password?`\n• I'll search our knowledge base first\n• If I can't find relevant info, I'll use my general knowledge\n• I always cite my sources when using the knowledge base"}},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn", "text": "*Examples:*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "• How do I reset my password?\n• What are the system requirements?\n• I'm having trouble with the installation\n• How do I configure the firewall settings?"}},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn", "text": "*Commands:*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "• `@YourBot help` - Get help\n• `@YourBot escalate` - Escalate to human agent\n• `@YourBot hello` - Start a conversation"}},
                {"type": "context", "elements": [{"type": "mrkdwn", "text": "Powered by AI | Knowledge Base + General Intelligence"}]}
            ]
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
    print(f"🔎 Searching knowledge base for: '{query}'")
    search_results = await search_knowledge_base(query)
    
    # Log the search results
    print("\n📚 KNOWLEDGE BASE SEARCH RESULTS:")
    print("-"*80)
    if "No relevant information found" in search_results:
        print("❌ No relevant information found in the knowledge base")
        has_relevant_info = False
    else:
        print("✅ Found relevant information in the knowledge base")
        # Show a preview of the results (first 200 chars)
        preview = search_results[:200] + "..." if len(search_results) > 200 else search_results
        print(f"Preview: {preview}")
        has_relevant_info = True
    print("-"*80)
    
    history_str = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history[-6:]])  # Only use last 6 exchanges
    
    # Construct the prompt based on whether we have relevant info
    if has_relevant_info:
        # Knowledge base prompt, which requires citing the provided source
        prompt = f"""
        You are a world-class expert support assistant for KaizenDev.
        Your goal is to provide a detailed answer based ONLY on the provided context from the knowledge base.

        **Instructions:**
        1.  Synthesize the 'Knowledge Base Context' to formulate your answer. Do not use outside knowledge.
        2.  Format your answer with clear headings and bullet points for readability.
        3.  At the end of your answer, you MUST include a "References" section with the URLs provided in the context.
        4.  If the context doesn't fully answer the question, acknowledge this limitation.
        5.  Be helpful and professional in your response.

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
    else:
        # General knowledge prompt, without asking for URLs
        prompt = f"""
        You are a world-class expert support assistant for KaizenDev.
        The user <@{user_id}> has asked a question for which there is no specific information in your internal knowledge base.
        Provide a helpful and comprehensive answer based on your own expertise.

        **Instructions:**
        1.  Provide a helpful and accurate response based on your general knowledge.
        2.  Format your answer with clear headings and bullet points for readability.
        3.  Acknowledge that this information is not from your knowledge base.
        4.  Suggest they escalate to a human agent if they need more specific information.
        5.  Be helpful and professional in your response.
        6.  DO NOT include any references or citations since this is general knowledge.

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
    
    # Log the prompt being sent to OpenAI
    print("\n📤 PROMPT SENT TO OPENAI:")
    print("-"*80)
    # Show a preview of the prompt (first 500 chars)
    prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
    print(f"Prompt preview: {prompt_preview}")
    print(f"Full prompt length: {len(prompt)} characters")
    print("-"*80)
    
    try:
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful expert assistant. Format your responses for readability in Slack with appropriate use of bolding, bullet points, and sections."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        answer = response.choices[0].message.content.strip()
        
        # Log the response from OpenAI
        print("\n📥 RESPONSE FROM OPENAI:")
        print("-"*80)
        # Show a preview of the response (first 300 chars)
        response_preview = answer[:300] + "..." if len(answer) > 300 else answer
        print(f"Response preview: {response_preview}")
        print(f"Full response length: {len(answer)} characters")
        print("-"*80)
        
        # Format for Slack
        formatted_answer = format_for_slack(answer, has_relevant_info)
        
        return formatted_answer
    except Exception as e:
        logger.error(f"Error in reasoning engine: {e}")
        print(f"\n❌ ERROR IN REASONING ENGINE: {e}")
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "My brain had a short circuit. :zap:"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "Can you try asking that again?"
                        }
                    ]
                }
            ]
        }

def format_for_slack(text: str, has_relevant_info: bool) -> dict:
    """Format text for better readability in Slack with proper blocks."""
    # Convert markdown bold to Slack bold
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    
    # Convert markdown headers to Slack bold
    text = re.sub(r'^### (.*?)$', r'*\1*', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'*\1*', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', r'*\1*', text, flags=re.MULTILINE)
    
    # Ensure bullet points are formatted correctly
    text = re.sub(r'^- (.*?)$', r'• \1', text, flags=re.MULTILINE)
    
    # Split the text into sections
    sections = text.split('\n\n')
    blocks = []
    
    # Process each section
    for section in sections:
        if section.strip():
            # Check if it's a references section - only include if we have relevant info
            if (section.strip().startswith("References:") or section.strip().startswith("*References:*")) and has_relevant_info:
                # Format references as a separate section with better formatting
                refs_text = section.replace("References:", "").replace("*References:*", "").strip()
                
                # Add a divider before references
                blocks.append({"type": "divider"})
                
                # Add references header
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Sources:*"
                    }
                })
                
                # Process each reference line
                ref_lines = refs_text.split('\n')
                for ref_line in ref_lines:
                    ref_line = ref_line.strip()
                    if ref_line:
                        # Extract URL if present
                        url_match = re.search(r'(https?://[^\s]+)', ref_line)
                        if url_match:
                            url = url_match.group(1)
                            # Create a clean title from the URL
                            title = ref_line.replace(url, "").strip()
                            if not title:
                                # If no title, use the URL path as title
                                parsed_url = re.sub(r'https?://', '', url)
                                title = parsed_url.split('/')[0]
                            
                            # Format as a button-like element
                            blocks.append({
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"• *{title}*\n<{url}|{url}>"
                                }
                            })
                        else:
                            # Regular text reference
                            blocks.append({
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"• {ref_line}"
                                }
                            })
            elif not (section.strip().startswith("References:") or section.strip().startswith("*References:*")):
                # Regular content section (not references)
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": section.strip()
                    }
                })
    
    # Add footer if no relevant info
    if not has_relevant_info:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "💡 *Tip:* For more specific information, consider escalating to a human agent."
                }
            ]
        })
    
    return {"blocks": blocks}

async def search_knowledge_base(query: str):
    # Increased threshold for better relevance - more strict
    RELEVANCE_THRESHOLD = 0.5  # Lowered threshold for more strict matching
    try:
        print(f"🔍 Creating embedding for query: '{query}'")
        response = await asyncio.to_thread(openai.embeddings.create, input=query, model="text-embedding-ada-002")
        query_vector = response.data[0].embedding
        
        print(f"🔍 Searching database with relevance threshold: {RELEVANCE_THRESHOLD}")
        results_df = await asyncio.to_thread(table.search(query_vector).limit(5).to_pandas)
        relevant_results = results_df[results_df['_distance'] <= RELEVANCE_THRESHOLD]
        
        print(f"🔍 Found {len(relevant_results)} relevant results out of {len(results_df)} total results")
        
        if relevant_results.empty:
            return "No relevant information found in the knowledge base."
        
        # Additional check: verify if the results are actually relevant to the query
        # This is important for cases where the vector search returns results that aren't semantically relevant
        print(f"🔍 Checking if results are actually relevant to the query...")
        
        # Create a prompt to check relevance
        relevance_check_prompt = f"""
        Given the user query: "{query}"
        
        And the following search results from our knowledge base:
        {relevant_results['text'].str.cat(sep=' ')}
        
        Determine if these search results contain information that directly answers the user's query.
        Respond with only "YES" if the results are relevant, or "NO" if they are not relevant.
        """
        
        try:
            relevance_response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a relevance checker. Respond with only YES or NO."},
                    {"role": "user", "content": relevance_check_prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            is_relevant = relevance_response.choices[0].message.content.strip().upper()
            print(f"🔍 Relevance check result: {is_relevant}")
            
            if is_relevant != "YES":
                print("❌ Search results are not actually relevant to the query")
                return "No relevant information found in the knowledge base."
        except Exception as e:
            print(f"⚠️ Error in relevance check: {e}")
            # If relevance check fails, continue with the results
        
        # Group by source URL to avoid duplication
        grouped_results = {}
        for _, row in relevant_results.iterrows():
            url = row['source_url']
            if url not in grouped_results:
                grouped_results[url] = {
                    'title': row['page_title'],
                    'content': row['text']
                }
            else:
                # Append content if we already have this URL
                grouped_results[url]['content'] += "\n\n" + row['text']
        
        # Format the results
        context_list = []
        for url, data in grouped_results.items():
            context_list.append(f"Source: {url}\nTitle: {data['title']}\nContent: {data['content']}")
        
        return "\n\n---\n\n".join(context_list)
    except Exception as e:
        logger.error(f"Error during knowledge base search: {e}")
        print(f"❌ ERROR IN KNOWLEDGE BASE SEARCH: {e}")
        return "No relevant information found in the knowledge base."

async def classify_intent(text: str):
    prompt = f"""
    You are an expert at classifying user intent. Classify the message into ONE of these categories: 'question', 'escalation', 'follow_up', 'greeting', or 'other'.
    Your response must be a single word.

    Examples:
    Message: "how do I reset my password?" -> Intent: question
    Message: "I have ZCC installed and my application is slowing down. what might be the cause?" -> Intent: question
    Message: "I need to speak to a manager now!" -> Intent: escalation
    Message: "thanks, that worked" -> Intent: follow_up
    Message: "hello there" -> Intent: greeting
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
                {"role": "system", "content": "You are an expert at classifying user intent. Respond with only a single word: question, escalation, follow_up, greeting, or other."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        intent = response.choices[0].message.content.strip().lower()
        intent = ''.join(c for c in intent if c.isalpha())
        
        valid_intents = ["question", "escalation", "follow_up", "greeting", "other"]
        return intent if intent in valid_intents else "other"
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        return "other"

# --- MAIN EXECUTION ---
async def main():
    logger.info("Starting KaizenDev Support Bot...")
    print("\n" + "="*80)
    print("🚀 STARTING KAIZENDEV SUPPORT BOT")
    print("="*80)
    print("Bot is now running. Mention the bot in Slack to ask questions.")
    print("Detailed logs will be shown here for each query processed.")
    print("="*80 + "\n")
    
    handler = AsyncSocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    await handler.start_async()

if __name__ == "__main__":
    asyncio.run(main())