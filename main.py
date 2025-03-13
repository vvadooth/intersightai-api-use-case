import os
import json
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to parse OpenAPI spec into paths
def get_openapi_spec_paths(spec_file_path):
    print("Parsing OpenAPI specification...")
    with open(spec_file_path, 'r') as f:
        specification = json.load(f)
    
    paths = []
    for p in specification["paths"]:
        for m in specification["paths"][p]:
            path_data = specification["paths"][p][m]
            path_data["method"] = m
            path_data["path"] = p
            paths.append(path_data)
    
    return paths

# Function to generate embeddings
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Using the model you specified
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

# Set up the database schema
def setup_database():
    print("Setting up database schema...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        DROP TABLE IF EXISTS intersight_openapi_kb; -- Remove this line if you want to append instead of recreate
        CREATE TABLE intersight_openapi_kb (
            id SERIAL PRIMARY KEY,
            path TEXT NOT NULL,
            method TEXT NOT NULL,
            content JSONB NOT NULL,
            metadata JSONB,
            embedding VECTOR(1536)
        );
        CREATE INDEX intersight_openapi_kb_embedding_idx 
        ON intersight_openapi_kb 
        USING hnsw (embedding vector_cosine_ops);
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database schema set up successfully!")

# Populate the database one endpoint at a time
def populate_database(paths):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    total_endpoints = len(paths)
    for i, path_data in enumerate(paths, 1):
        print(f"Processing endpoint {i}/{total_endpoints}: {path_data['method'].upper()} {path_data['path']}")
        
        # Use description or summary for embedding, fallback to full content
        text_to_embed = path_data.get("description", path_data.get("summary", json.dumps(path_data)))
        try:
            embedding = get_embedding(text_to_embed)
            print(f"Generated embedding for {path_data['method'].upper()} {path_data['path']}")
        except Exception as e:
            print(f"Skipping {path_data['method'].upper()} {path_data['path']} due to embedding error: {e}")
            continue
        
        try:
            cur.execute("""
                INSERT INTO intersight_openapi_kb (path, method, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s);
            """, (
                path_data["path"],
                path_data["method"],
                json.dumps(path_data),
                json.dumps({
                    "tags": path_data.get("tags", []),
                    "summary": path_data.get("summary", "")
                }),
                embedding
            ))
            conn.commit()
            print(f"Successfully inserted {path_data['method'].upper()} {path_data['path']} into database")
        except Exception as e:
            print(f"Error inserting {path_data['method'].upper()} {path_data['path']} into database: {e}")
            conn.rollback()  # Roll back on error to keep connection alive
    
    cur.close()
    conn.close()
    print(f"Finished processing {total_endpoints} endpoints!")

# Main execution
if __name__ == "__main__":
    # Path to your OpenAPI spec
    spec_file_path = "intersight-openapi.json"
    
    # Set up database
    setup_database()
    
    # Parse the spec
    paths = get_openapi_spec_paths(spec_file_path)
    print(f"Extracted {len(paths)} endpoints from the OpenAPI spec.")
    
    # Populate the database one by one
    populate_database(paths)

    