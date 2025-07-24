"""
Test super verbose per debugging
"""
import json
import requests
import time
from datetime import datetime
import os

# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

print("🔧 VERBOSE EMOTIONAL INTELLIGENCE TEST")
print("=" * 70)
print(f"📍 API Endpoint: {API_URL}")
print(f"🔑 API Key: {API_KEY[:20]}...{API_KEY[-4:]}")
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Test solo un modello per vedere tutto
model_name = "Mixtral-8x22B"
model_id = "mistralai/mixtral-8x22b-instruct"

prompt = """Identify ALL emotions in this text:
"I can't believe I have to work late again tonight. I promised my daughter I'd be at her recital."
List each emotion you detect."""

print(f"\n🤖 Testing: {model_name}")
print(f"📝 Model ID: {model_id}")
print(f"📄 Prompt: {prompt[:50]}...")
print("-" * 70)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/lodetomasi/emotional-intelligence-llm",
    "X-Title": "EI Test Debug"
}

data = {
    "model": model_id,
    "messages": [
        {
            "role": "system",
            "content": "You are participating in an emotional intelligence test."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    "temperature": 0.7,
    "max_tokens": 500
}

print("\n📤 REQUEST DETAILS:")
print(f"Headers: {json.dumps(headers, indent=2)}")
print(f"Data: {json.dumps(data, indent=2)}")

print("\n🌐 Making API call...")
start_time = time.time()

try:
    response = requests.post(
        API_URL,
        headers=headers,
        json=data,
        timeout=30
    )
    
    elapsed = time.time() - start_time
    print(f"⏱️  Response received in {elapsed:.2f} seconds")
    print(f"📊 Status Code: {response.status_code}")
    print(f"📋 Response Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS!")
        print(f"📤 Full Response:")
        print(json.dumps(result, indent=2))
        
        ai_response = result["choices"][0]["message"]["content"]
        print(f"\n🧠 AI Response:")
        print("-" * 50)
        print(ai_response)
        print("-" * 50)
        
        print(f"\n💰 Token Usage:")
        print(f"   Prompt tokens: {result.get('usage', {}).get('prompt_tokens', 'N/A')}")
        print(f"   Completion tokens: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
        print(f"   Total tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}")
        
    else:
        print(f"\n❌ ERROR!")
        print(f"Response body: {response.text}")
        
except Exception as e:
    print(f"\n❌ EXCEPTION!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    print(f"Traceback:")
    traceback.print_exc()

print("\n✅ Test completed!")
print("=" * 70)