import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from pymongo import MongoClient
import sqlite3
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import duckdb
import oracledb
import os
import redis
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="Database Chatbot", layout="wide",page_icon="🛢️")

st.title("🧠 Multi Database Chatbot Interface")
st.markdown("Ask questions about your database in natural language.")

# Initialize LLM at the top
llm = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))
db_category = st.radio("Select Database Type", ["SQL Databases", "NoSQL Databases"])
if db_category =="SQL Databases":

    def connection_string_type(db_choice):
        if db_choice == "PostgreSQL":
            return "postgresql://user:password@localhost/dbname"
        elif db_choice == "MySQL":
            return "mysql://user:password@localhost/dbname"
        elif db_choice == "SQLite":
            return "sqlite:///dbname.db"
        elif db_choice == "Oracle":
            return "oracle+cx_oracle://user:password@localhost:1521/dbname"
        elif db_choice == "DuckDB":
            return "duckdb:///path/to/database.duckdb"
        else:
            return None


    # Initialize session state for database connection
    if "db_choice" not in st.session_state:
        st.session_state["db_choice"] = None
        
    if "connection_string" not in st.session_state:
        st.session_state["connection_string"] = None

    if "db_path" not in st.session_state:
        st.session_state["db_path"] = None

    


    # ask user to choose a database
    st.session_state["db_choice"]  = st.sidebar.selectbox(
        "Choose a database",
        ("PostgreSQL", "MySQL", "SQLite", "Oracle", "DuckDB"),
    )


    if st.session_state["db_choice"] is None:
        st.error("Please select a database to continue.")
        st.stop()

    # add line break
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Database Connection")



        
        
    # ask for connection string
    if st.session_state["db_choice"] == "SQLite":
        st.session_state["db_path"] = st.sidebar.text_input(
            "Enter your SQLite database file path:",
            placeholder=f"e.g. /path/to/your/database.db"
        )
    st.session_state["connection_string"] = st.sidebar.text_input(
            "Enter your database connection string:",
            placeholder=f"e.g. {connection_string_type(st.session_state['db_choice'])}"
    )



    if not st.session_state["connection_string"]:
            st.sidebar.error("Please enter a valid connection string.")
            st.warning("You need to provide a connection string to connect to the database.")
            st.stop()
    
    st.sidebar.success(f"You selected: {st.session_state['db_choice']}")


    with st.expander("Database Connection Details", expanded=False):
        st.write(f"**Database Type:** {st.session_state['db_choice']}")
        st.write(f"**Connection String:** {st.session_state['connection_string']}")
        st.write(f"**Model:** ChatGroq - Gemma2-9b-It")
        
        
    with st.spinner("Setting up the db connection..."):
        if st.session_state["db_choice"] == "SQLite":
            # For SQLite, we use the file path directly
            db_path = st.session_state["db_path"]
            if not db_path:
                st.error("Please provide a valid SQLite database file path.")
                st.stop()
            creator = lambda: sqlite3.connect(db_path)
            db = SQLDatabase(create_engine("sqlite:///", creator=creator))
            
        else:
            # For other databases, we use the connection string
            connection_string = st.session_state["connection_string"]
            if not connection_string:
                st.error("Please provide a valid connection string.")
                st.stop()
            db = SQLDatabase(create_engine(connection_string))
            
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
        
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

                

        
    # Ask user to input a query
    user_query = st.chat_input(placeholder="Ask anything from the database")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback=StreamlitCallbackHandler(st.container())
            response=agent.run(user_query,callbacks=[streamlit_callback])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write(response)
elif db_category == "NoSQL Databases":
    nosql_type = st.sidebar.selectbox("Choose NoSQL DB Type", ["MongoDB", "Redis"])
    
    if nosql_type == "MongoDB":
        mongo_uri = st.sidebar.text_input("MongoDB URI (e.g. mongodb+srv://<user>:<pass>@cluster.mongodb.net/)")
        db_name = st.sidebar.text_input("Database Name")
        collection_name = st.sidebar.text_input("Collection Name")

        if mongo_uri and db_name and collection_name:
            try:
                client = MongoClient(mongo_uri)
                collection = client[db_name][collection_name]

                if "nosql_messages" not in st.session_state:
                    st.session_state.nosql_messages = []

                # Display previous NoSQL messages
                for msg in st.session_state.nosql_messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                # Input from user
                user_query = st.chat_input("Ask your NoSQL database 💬")
                if user_query:
                    st.chat_message("user").write(user_query)
                    st.session_state.nosql_messages.append({"role": "user", "content": user_query})

                    with st.chat_message("assistant"):
                        with st.spinner("Generating MongoDB query..."):
                            mongo_prompt = f"""
You're a MongoDB expert. Convert the user's request into a MongoDB `.find()` query.
Respond ONLY with a valid **Python dictionary** representing the query (e.g. {{'age': {{'$gt': 25}}}}). Do not explain.

User Request: "{user_query}"
                            """.strip()

                            # Get the MongoDB query as string from LLM
                            response = llm.invoke(mongo_prompt).content.strip()

                            # Try safe evaluation
                            import ast
                            try:
                                mongo_query = ast.literal_eval(response)
                                docs = list(collection.find(mongo_query).limit(10))

                                st.session_state.nosql_messages.append({
                                    "role": "assistant",
                                    "content": f"```json\n{docs}\n```"
                                })

                                st.write("🔍 Results:")
                                st.json(docs)
                            except Exception as e:
                                error_msg = f"❌ Failed to parse or run query: {e}"
                                st.session_state.nosql_messages.append({"role": "assistant", "content": error_msg})
                                st.error(error_msg)

            except Exception as conn_err:
                st.error(f"❌ Could not connect to MongoDB: {conn_err}")
    elif nosql_type == "Redis":
        redis_uri = st.sidebar.text_input("Redis URI (e.g., redis://:password@host:6379/0)")
        
        if redis_uri:
            try:
                r = redis.StrictRedis.from_url(redis_uri, decode_responses=True)
                st.success("✅ Connected to Redis!")

                # Show Redis info in sidebar
                with st.sidebar:
                    st.markdown("### Redis Info")
                    try:
                        info = r.info()
                        st.write(f"🔢 Total Keys: {r.dbsize()}")
                        st.write(f"💾 Used Memory: {info.get('used_memory_human', 'N/A')}")
                    except:
                        pass

                if "redis_messages" not in st.session_state:
                    st.session_state.redis_messages = []

                for msg in st.session_state.redis_messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                user_query = st.chat_input("Ask about your Redis data in natural language 💬 (e.g., 'show all keys', 'get user data', 'find keys with pattern user*')")

                if user_query:
                    st.chat_message("user").write(user_query)
                    st.session_state.redis_messages.append({"role": "user", "content": user_query})

                    with st.chat_message("assistant"):
                        with st.spinner("🧠 Processing your request..."):
                            try:
                                # Get sample keys for context
                                sample_keys = r.keys("*")[:10] if r.dbsize() > 0 else []
                                
                                # Create prompt for LLM to convert natural language to Redis commands
                                redis_prompt = f"""
You are a Redis expert. Convert the user's natural language request into Redis commands and execute them.

Available Redis commands: GET, SET, KEYS, DEL, INCR, DECR, HGET, HSET, HGETALL, LRANGE, SADD, SMEMBERS, etc.

Sample keys in database: {sample_keys[:5]}
Database size: {r.dbsize()} keys

User request: "{user_query}"

Respond with a JSON object containing:
1. "explanation": Brief explanation of what you're doing
2. "commands": List of Redis commands to execute
3. "reasoning": Why you chose these commands

Example response:
{{
    "explanation": "Getting all keys that match the pattern",
    "commands": ["KEYS user:*"],
    "reasoning": "User wants to find user-related data"
}}

Only respond with valid JSON.
"""

                                # Get LLM response
                                response = llm.invoke(redis_prompt).content.strip()
                                
                                # Parse LLM response
                                import json
                                try:
                                    parsed_response = json.loads(response)
                                    explanation = parsed_response.get("explanation", "Executing Redis commands...")
                                    commands = parsed_response.get("commands", [])
                                    reasoning = parsed_response.get("reasoning", "")

                                    st.write(f"🤖 **Analysis:** {explanation}")
                                    if reasoning:
                                        st.write(f"💭 **Reasoning:** {reasoning}")

                                    results = []
                                    
                                    for cmd in commands:
                                        try:
                                            # Parse and execute Redis command
                                            parts = cmd.split()
                                            if not parts:
                                                continue
                                                
                                            cmd_name = parts[0].upper()
                                            args = parts[1:]

                                            if cmd_name == "KEYS":
                                                result = r.keys(*args)
                                            elif cmd_name == "GET":
                                                result = r.get(*args)
                                            elif cmd_name == "HGETALL":
                                                result = r.hgetall(*args)
                                            elif cmd_name == "HGET":
                                                result = r.hget(*args)
                                            elif cmd_name == "LRANGE":
                                                # Default to get all list items
                                                if len(args) == 1:
                                                    args.extend([0, -1])
                                                result = r.lrange(*args)
                                            elif cmd_name == "SMEMBERS":
                                                result = list(r.smembers(*args))
                                            elif cmd_name == "TYPE":
                                                result = r.type(*args).decode('utf-8') if isinstance(r.type(*args), bytes) else r.type(*args)
                                            elif cmd_name == "TTL":
                                                result = r.ttl(*args)
                                            elif cmd_name == "EXISTS":
                                                result = r.exists(*args)
                                            else:
                                                result = f"⚠️ Command '{cmd_name}' not supported in this interface"

                                            results.append({
                                                "command": cmd,
                                                "result": result
                                            })

                                        except Exception as cmd_error:
                                            results.append({
                                                "command": cmd,
                                                "error": str(cmd_error)
                                            })

                                    # Display results
                                    st.write("### 📊 Results:")
                                    for i, res in enumerate(results, 1):
                                        with st.expander(f"Command {i}: `{res['command']}`", expanded=True):
                                            if "error" in res:
                                                st.error(f"❌ Error: {res['error']}")
                                            else:
                                                result = res["result"]
                                                if isinstance(result, (list, dict)):
                                                    st.json(result)
                                                elif result is None:
                                                    st.write("🔍 No data found")
                                                else:
                                                    st.write(f"**Result:** {result}")

                                    # Save to chat history
                                    response_text = f"Executed {len(commands)} command(s). Found {len([r for r in results if 'error' not in r])} successful results."
                                    st.session_state.redis_messages.append({"role": "assistant", "content": response_text})

                                except json.JSONDecodeError:
                                    # Fallback: try to extract commands from response
                                    st.write("🤖 **Processing your request...**")
                                    
                                    # Simple keyword-based command generation
                                    query_lower = user_query.lower()
                                    
                                    if "all keys" in query_lower or "show keys" in query_lower:
                                        keys = r.keys("*")
                                        st.write(f"📋 **All Keys ({len(keys)}):**")
                                        if keys:
                                            for key in keys[:20]:  # Show first 20 keys
                                                key_type = r.type(key).decode('utf-8') if isinstance(r.type(key), bytes) else r.type(key)
                                                st.write(f"• `{key}` (type: {key_type})")
                                            if len(keys) > 20:
                                                st.write(f"... and {len(keys) - 20} more keys")
                                        else:
                                            st.write("No keys found in database")
                                    
                                    elif "get" in query_lower and any(key in query_lower for key in ["user", "data", "info"]):
                                        # Try to find and get user-related keys
                                        user_keys = r.keys("*user*") + r.keys("*User*")
                                        if user_keys:
                                            st.write("👤 **User-related data found:**")
                                            for key in user_keys[:5]:
                                                value = r.get(key)
                                                st.write(f"• `{key}`: {value}")
                                        else:
                                            st.write("No user-related keys found")
                                    
                                    else:
                                        # Generic response
                                        st.write("🔍 **Database Overview:**")
                                        total_keys = r.dbsize()
                                        st.write(f"• Total keys: {total_keys}")
                                        
                                        if total_keys > 0:
                                            sample_keys = r.keys("*")[:5]
                                            st.write("• Sample keys:")
                                            for key in sample_keys:
                                                key_type = r.type(key).decode('utf-8') if isinstance(r.type(key), bytes) else r.type(key)
                                                st.write(f"  - `{key}` (type: {key_type})")

                            except Exception as e:
                                error_msg = f"❌ Failed to process request: {e}"
                                st.session_state.redis_messages.append({"role": "assistant", "content": error_msg})
                                st.error(error_msg)

            except Exception as e:
                st.error(f"❌ Could not connect to Redis: {e}")


