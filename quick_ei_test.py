import json
import requests
import time
from datetime import datetime
import os

# Configurazione
API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Modelli da testare
MODELS = {
    "Mixtral-8x22B": "mistralai/mixtral-8x22b-instruct",
    "Claude Opus 4": "anthropic/claude-opus-4",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct"
}

# Test scenario semplice
prompt = """Identifica TUTTE le emozioni in questo testo:

"Non posso credere di dover lavorare fino a tardi di nuovo. Ho promesso a mia figlia che sarei stato al suo saggio, ma questa scadenza è inderogabile. Mi sento come se la stessi sempre deludendo."

Elenca ogni emozione e spiega perché."""

print("🧠 Test Rapido Intelligenza Emotiva")
print("=" * 50)

for model_name, model_id in MODELS.items():
    print(f"\nTesting {model_name}...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            print(f"✅ Successo!")
            print(f"Risposta: {ai_response[:200]}...")
            
            # Salva risultato
            with open(f"{model_name.replace(' ', '_')}_result.txt", "w", encoding="utf-8") as f:
                f.write(ai_response)
        else:
            print(f"❌ Errore: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Errore: {e}")
    
    time.sleep(2)

print("\n✅ Test completato!")