import os
import streamlit as st
import psycopg2
import json
import re
import logging
from openai import OpenAI
from urllib.parse import urlencode

# Set up logging
logging.basicConfig(
    filename='openai_requests.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to generate embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Query the knowledge base with a limit
def query_kb(query_text, limit=50):
    query_embedding = get_embedding(query_text)
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT path, method, metadata->>'summary' AS summary, content
        FROM intersight_openapi_kb
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, limit))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# Clean and fix incomplete JSON
def clean_json_response(raw_response):
    cleaned = re.sub(r'```json|```', '', raw_response).strip()
    if not cleaned.endswith(']'):
        cleaned += ']'
    return cleaned

# Generate cURL command from API call details
def generate_curl_command(call):
    method = call["method"]
    path = call["path"]
    headers = call.get("headers", {})
    query_params = call.get("query_params", None)
    body = call.get("body", None)
    
    # Base URL (replace with your actual Intersight base URL if needed)
    base_url = "https://intersight.com"
    url = f"{base_url}{path}"
    
    # Add query parameters if present
    if query_params:
        url += "?" + urlencode(query_params)
    
    # Build cURL command
    curl = f"curl -X {method} \"{url}\""
    for header, value in headers.items():
        curl += f" -H \"{header}: {value}\""
    if body:
        curl += f" -d '{json.dumps(body)}'"
    
    return curl

# Generate follow-up queries using ChatGPT
def generate_followup_queries(original_query, initial_results):
    api_data = "\n".join([f"{method.upper()} {path}: {summary}\nContent: {content}" 
                          for path, method, summary, content in initial_results])
    prompt = f"""
    Given the original query: "{original_query}"
    And this set of 50 best-aligned API endpoints from the Cisco Intersight OpenAPI spec:
    {api_data}
    
    Generate 5 more specific queries I can use to perform vector search to retrieve 15 additional relevant API calls.
    Return the queries as a valid JSON list, e.g., ["query1", "query2", "query3", "query4", "query5"].
    Ensure the response is strictly JSON and nothing else.
    """
    
    logger.info("Sending to OpenAI (generate_followup_queries):\n%s", prompt)
    st.write("Prompt sent to OpenAI (follow-up queries):", prompt)
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        raw_response = completion.choices[0].message.content
        logger.info("Received from OpenAI (generate_followup_queries):\n%s", raw_response)
        st.write(f"Raw response from OpenAI (follow-up queries): {raw_response}")
        
        cleaned_response = clean_json_response(raw_response)
        queries = json.loads(cleaned_response)
        if not isinstance(queries, list) or len(queries) != 5:
            raise ValueError("Response is not a list of 5 queries")
        return queries
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse OpenAI response as JSON: {e}")
        st.write(f"Raw response: {raw_response}")
        return [
            f"{original_query} details",
            f"{original_query} related endpoints",
            f"{original_query} parameters",
            f"{original_query} security",
            f"{original_query} examples"
        ]
    except Exception as e:
        st.error(f"Error generating follow-up queries: {e}")
        return [
            f"{original_query} details",
            f"{original_query} related endpoints",
            f"{original_query} parameters",
            f"{original_query} security",
            f"{original_query} examples"
        ]

# Analyze APIs and determine required calls
def analyze_api_calls(original_query, initial_results, additional_results):
    initial_api_data = "\n".join([f"{method.upper()} {path}: {summary}\nContent: {content}" 
                                  for path, method, summary, content in initial_results])
    additional_api_data = "\n".join([f"{method.upper()} {path}: {summary}\nContent: {content}" 
                                     for path, method, summary, content in additional_results])
    
    prompt = f"""
    Original query: "{original_query}"
    
    Initial 50 API calls retrieved:
    {initial_api_data}
    
    Follow-up queries and additional 15 API calls retrieved:
    {additional_api_data}
    
    Based on the original query, determine the minimum number of API calls needed to fully answer it (one or multiple).
    - If only one call is sufficient, select the single most relevant API call.
    - If multiple calls are required, explain why and select only the necessary ones.
    - Use the provided summaries and content (e.g., parameters, responses) to justify your choice(s).
    
    For each required API call, provide:
    - The HTTP method and path (e.g., "GET /api/v1/example")
    - The required headers (e.g., Authorization, Content-Type)
    - Any required query parameters or request body (if applicable)
    
    Structure the response as a valid JSON object with a list of API call details, e.g.,
    [
        {{
            "method": "GET",
            "path": "/api/v1/example",
            "headers": {{"Authorization": "Bearer <token>", "Content-Type": "application/json"}},
            "query_params": {{"filter": "example"}},
            "body": null
        }}
    ]
    Ensure the response is strictly JSON and nothing else.
    """
    
    logger.info("Sending to OpenAI (analyze_api_calls):\n%s", prompt)
    st.write("Prompt sent to OpenAI (analyze_api_calls):", prompt)
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        raw_response = completion.choices[0].message.content
        logger.info("Received from OpenAI (analyze_api_calls):\n%s", raw_response)
        st.write(f"Raw analysis response from OpenAI: {raw_response}")
        
        cleaned_response = clean_json_response(raw_response)
        api_calls = json.loads(cleaned_response)
        if not isinstance(api_calls, list):
            raise ValueError("Response is not a list of API calls")
        return api_calls
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse OpenAI analysis response as JSON: {e}")
        st.write(f"Raw response: {raw_response}")
        return []
    except Exception as e:
        st.error(f"Error analyzing API calls: {e}")
        return []

# Streamlit interface
st.title("Intersight OpenAPI Chatbot")
st.write("Ask questions about the Cisco Intersight API!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Enter your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        st.write("Retrieving 50 initial API endpoints...")
        initial_results = query_kb(prompt, limit=50)
        
        st.write("Generating follow-up queries...")
        followup_queries = generate_followup_queries(prompt, initial_results)
        st.write("Follow-up queries generated:", followup_queries)
        
        additional_results = []
        for query in followup_queries:
            results = query_kb(query, limit=3)
            additional_results.extend(results)
        
        st.write("Analyzing API calls...")
        api_calls = analyze_api_calls(prompt, initial_results, additional_results)
        
        response = f"### Original Query: {prompt}\n\n"
        response += f"Based on the analysis, {len(api_calls)} API call(s) are required:\n\n"
        
        for call in api_calls:
            response += f"#### {call['method']} {call['path']}\n"
            response += "**Headers**:\n"
            for header, value in call.get('headers', {}).items():
                response += f"- {header}: {value}\n"
            if 'query_params' in call and call['query_params']:
                response += "**Query Parameters**:\n"
                for param, value in call['query_params'].items():
                    response += f"- {param}: {value}\n"
            if 'body' in call and call['body']:
                response += "**Request Body**:\n```json\n" + json.dumps(call['body'], indent=2) + "\n```\n"
            # Add cURL command in a cool code block
            curl_command = generate_curl_command(call)
            response += "**cURL Command**:\n"
            response += f"```bash\n{curl_command}\n```\n"
            response += "\n"
        
        if not api_calls:
            response += "Sorry, I couldn’t determine a specific API call to answer your query."
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})