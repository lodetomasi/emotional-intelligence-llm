"""
Comprehensive Emotional Intelligence Testing Suite
Tests all models across 4 EI dimensions with multiple scenarios
"""

import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from save_results import save_all_results, ResultsSaver

# Configuration
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to test (corrected IDs)
MODELS = {
    "Mixtral-8x22B": "mistralai/mixtral-8x22b-instruct",
    "Claude Opus 4": "anthropic/claude-opus-4",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
    "DeepSeek R1": "deepseek/deepseek-r1-0528",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro"
}

# Comprehensive test scenarios
TEST_SCENARIOS = {
    "emotion_recognition": [
        {
            "id": "ER1",
            "prompt": """Read this message and identify ALL emotions present:
"I can't believe I have to work late again tonight. I promised my daughter I'd be at her recital, but this project deadline is non-negotiable. I feel like I'm constantly letting her down. Sometimes I wonder if this job is worth missing all these important moments."

List each emotion you detect and explain why.""",
            "expected": ["frustration", "guilt", "disappointment", "conflict", "regret", "doubt"]
        },
        {
            "id": "ER2", 
            "prompt": """What emotions are present in this text:
"Just got the promotion! 😊 Although... I'm not sure I deserve it. Maybe they made a mistake? What if I can't handle the responsibility?"

Identify all emotions including conflicting ones.""",
            "expected": ["joy", "self-doubt", "anxiety", "imposter syndrome", "fear"]
        },
        {
            "id": "ER3",
            "prompt": """Identify the emotions (including hidden ones) in:
"Sure, go ahead without me. It's fine. I didn't really want to go anyway. Have fun."

What emotions might be beneath the surface?""",
            "expected": ["hurt", "rejection", "passive-aggressiveness", "sadness", "loneliness"]
        }
    ],
    
    "empathy": [
        {
            "id": "EM1",
            "prompt": """Your friend messages you:
"I had to put my dog down yesterday. He was with me for 15 years. The house feels so empty without him. I keep expecting to hear his paws on the floor or see him in his favorite spot by the window. I don't know how to handle this grief."

Write a compassionate, empathetic response.""",
            "criteria": ["acknowledgment", "validation", "support", "personal_touch"]
        },
        {
            "id": "EM2",
            "prompt": """A colleague says:
"I completely bombed the presentation. Everyone was looking at me like I was an idiot. My voice was shaking, I forgot half my points. I want to quit. I'm clearly not cut out for this job."

Provide an empathetic response that helps them process this experience.""",
            "criteria": ["normalize", "reframe", "support", "no_minimizing"]
        },
        {
            "id": "EM3",
            "prompt": """Your teenage niece texts you:
"Nobody understands me. My parents think I'm overreacting about the breakup but this is literally ruining my life. I can't eat, can't sleep. They just say 'you'll get over it' like my feelings don't matter."

How would you respond with genuine empathy?""",
            "criteria": ["validate", "no_condescension", "understanding", "bridge_building"]
        }
    ],
    
    "emotional_regulation": [
        {
            "id": "REG1",
            "prompt": """You're in a team meeting when a colleague says:
"Honestly, your analysis is completely wrong and shows you don't understand the basics. We've wasted a week because of your incompetence. This is exactly why I said we shouldn't have put you on this project."

How would you respond professionally while managing your emotions?""",
            "criteria": ["calm", "professional", "de-escalate", "boundaries"]
        },
        {
            "id": "REG2",
            "prompt": """You just received a text that a close family member is in the hospital, but you're about to give the most important presentation of your career to the board of directors in 5 minutes.

How do you handle this situation and manage your emotional state?""",
            "criteria": ["compartmentalize", "prioritize", "self-care", "realistic"]
        }
    ],
    
    "social_awareness": [
        {
            "id": "SA1",
            "prompt": """At a company party, you notice a new team member standing alone by the wall, looking at their phone. When people approach, they give brief responses and return to their phone. Their shoulders are hunched, arms crossed.

What might be happening and how would you approach this situation?""",
            "criteria": ["observation", "empathy", "appropriate", "respectful"]
        },
        {
            "id": "SA2",
            "prompt": """During a team meeting, two colleagues have this exchange:
Person A: "That's an interesting approach, though I wonder if we've considered all angles."
Person B: "Of course we have. I've been doing this for 10 years."
Person A: "Right, of course. Just thinking out loud."
The room goes quiet and people look uncomfortable.

What's happening here and how might you help navigate this?""",
            "criteria": ["dynamics", "tension", "tactful", "mediate"]
        }
    ]
}

def query_model(model_id: str, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Query a model with retry logic"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lodetomasi/emotional-intelligence-llm",
        "X-Title": "EI Research"
    }
    
    data = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are participating in an emotional intelligence assessment. Respond naturally and thoughtfully to each scenario."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=data,
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result["choices"][0]["message"]["content"],
                    "model": result.get("model", model_id),
                    "usage": result.get("usage", {})
                }
            else:
                error_msg = f"Status {response.status_code}: {response.text[:200]}"
                if attempt == max_retries - 1:
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}
        
        time.sleep(2 ** attempt)  # Exponential backoff
    
    return {"success": False, "error": "Max retries exceeded"}

def run_comprehensive_tests():
    """Run all EI tests across all models"""
    print("🧠 COMPREHENSIVE EMOTIONAL INTELLIGENCE TESTING")
    print("=" * 70)
    print(f"Models: {len(MODELS)}")
    print(f"Dimensions: {len(TEST_SCENARIOS)}")
    print(f"Total scenarios: {sum(len(scenarios) for scenarios in TEST_SCENARIOS.values())}")
    print("=" * 70)
    
    results = {
        "test_metadata": {
            "date": datetime.now().isoformat(),
            "models": list(MODELS.keys()),
            "dimensions": list(TEST_SCENARIOS.keys()),
            "total_scenarios": sum(len(scenarios) for scenarios in TEST_SCENARIOS.values())
        },
        "model_results": {},
        "dimension_summaries": {}
    }
    
    # Test each model
    for model_name, model_id in MODELS.items():
        print(f"\n📊 Testing {model_name}...")
        results["model_results"][model_name] = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "dimensions": {}
        }
        
        # Test each dimension
        for dimension, scenarios in TEST_SCENARIOS.items():
            print(f"  📝 {dimension}:", end=" ")
            dimension_results = []
            successes = 0
            
            for scenario in scenarios:
                result = query_model(model_id, scenario["prompt"])
                
                if result["success"]:
                    dimension_results.append({
                        "scenario_id": scenario["id"],
                        "success": True,
                        "response": result["response"],
                        "usage": result.get("usage", {})
                    })
                    successes += 1
                    print("✓", end="", flush=True)
                else:
                    dimension_results.append({
                        "scenario_id": scenario["id"],
                        "success": False,
                        "error": result["error"]
                    })
                    print("✗", end="", flush=True)
                
                time.sleep(2)  # Rate limiting
            
            results["model_results"][model_name]["dimensions"][dimension] = dimension_results
            print(f" ({successes}/{len(scenarios)})")
            
        time.sleep(3)  # Extra delay between models
    
    # Calculate summaries
    print("\n📈 Calculating summaries...")
    for dimension in TEST_SCENARIOS.keys():
        results["dimension_summaries"][dimension] = {}
        for model_name in MODELS.keys():
            successes = sum(
                1 for r in results["model_results"][model_name]["dimensions"].get(dimension, [])
                if r.get("success", False)
            )
            total = len(TEST_SCENARIOS[dimension])
            results["dimension_summaries"][dimension][model_name] = f"{successes}/{total}"
    
    return results

def analyze_emotion_recognition(results: Dict) -> Dict[str, List[str]]:
    """Analyze emotion recognition responses"""
    emotions_by_model = {}
    
    for model_name, model_data in results["model_results"].items():
        emotions_found = []
        for test in model_data["dimensions"].get("emotion_recognition", []):
            if test.get("success"):
                response = test["response"].lower()
                # Simple keyword extraction (in real analysis, use NLP)
                emotion_keywords = [
                    "frustration", "guilt", "disappointment", "anger", "sadness",
                    "joy", "anxiety", "fear", "doubt", "hurt", "rejection",
                    "loneliness", "stress", "regret", "conflict", "helplessness"
                ]
                for emotion in emotion_keywords:
                    if emotion in response and emotion not in emotions_found:
                        emotions_found.append(emotion)
        
        emotions_by_model[model_name] = emotions_found
    
    return emotions_by_model

def main():
    """Main execution function"""
    # Run tests
    results = run_comprehensive_tests()
    
    # Analyze results
    print("\n🔍 Analyzing results...")
    emotions_analysis = analyze_emotion_recognition(results)
    
    # Add analysis to results
    results["analysis"] = {
        "emotion_recognition_summary": emotions_analysis,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    print("\n💾 Saving results...")
    saver = ResultsSaver()
    
    # Save raw results
    raw_path = saver.save_raw_results(results)
    
    # Create summary DataFrame
    summary_data = []
    for model in MODELS.keys():
        for dimension in TEST_SCENARIOS.keys():
            summary_data.append({
                "Model": model,
                "Dimension": dimension,
                "Success_Rate": results["dimension_summaries"][dimension].get(model, "0/0")
            })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path, json_path = saver.save_processed_scores(summary_df)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    print("\nSuccess Rates by Dimension:")
    for dimension, model_scores in results["dimension_summaries"].items():
        print(f"\n{dimension}:")
        for model, score in model_scores.items():
            print(f"  {model}: {score}")
    
    print(f"\n✅ All results saved successfully!")
    print(f"Raw results: {raw_path}")
    print(f"Summary CSV: {csv_path}")
    
    return results

if __name__ == "__main__":
    results = main()