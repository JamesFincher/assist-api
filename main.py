from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiosqlite
import openai
import os
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    user_id: str
    message: str
    max_tokens: int = 100
    n: int = 1
    temperature: float = 0.5

async def init_db():
    db = await aiosqlite.connect("chat_history.db")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    await db.commit()
    return db

@app.on_event("startup")
async def startup():
    app.state.db = await init_db()

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()

async def add_message(user_id, role, content):
    await app.state.db.execute(
        "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content),
    )
    await app.state.db.commit()

async def get_conversation_history(user_id):
    cursor = await app.state.db.execute(
        "SELECT role, content FROM chat_history WHERE user_id = ?",
        (user_id,),
    )
    rows = await cursor.fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]

@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        conversation_history = await get_conversation_history(request.user_id)
        
        if not conversation_history:
            # Add system message for the first time
            await add_message(request.user_id, "system", "You are a helpful assistant.")
            conversation_history.append({"role": "system", "content": "You are a helpful assistant."})

        # Add user message
        await add_message(request.user_id, "user", request.message)

        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            max_tokens=request.max_tokens,
            n=request.n,
            temperature=request.temperature,
        )

        responses = [choice.message.content.strip() for choice in chat_completion.choices]

        # Add assistant message
        await add_message(request.user_id, "assistant", responses[0])

        if len(responses) == 1:
            return {"response": responses[0]}
        else:
            return {"responses": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
async def history(user_id: str):
    try:
        conversation_history = await get_conversation_history(user_id)
        return {"conversation_history": conversation_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
