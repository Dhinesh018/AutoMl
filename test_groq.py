from groq import Groq
from src.config import GROQ_API_KEY

print("Testing Groq API connection...")

if not GROQ_API_KEY:
    print("❌ Error: No API key found!")
    exit(1)

try:
    client = Groq(api_key=GROQ_API_KEY)
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Say 'Hello, AutoML!' if you can hear me."}
        ],
        temperature=0.7,
        max_tokens=50
    )
    
    response_text = completion.choices[0].message.content
    print(f"✅ API Connection Successful!")
    print(f"Groq says: {response_text}")
    
except Exception as e:
    print(f"❌ Error: {e}")