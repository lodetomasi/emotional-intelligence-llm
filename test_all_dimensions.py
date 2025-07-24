"""
Complete Emotional Intelligence Test - All 4 Dimensions
Run this script to test all models across all EI dimensions
"""

import json
import requests
import time
from datetime import datetime
import os

# Configuration - Use your API key
API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to test
MODELS = {
    "Mixtral-8x22B": "mistralai/mixtral-8x22b-instruct",
    "Claude Opus 4": "anthropic/claude-opus-4",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
    "DeepSeek R1": "deepseek/deepseek-r1-0528",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro"
}

# Comprehensive test scenarios for all 4 dimensions
TEST_SCENARIOS = {
    "emotion_recognition": [
        {
            "id": "ER1",
            "prompt": """Identify ALL emotions in this text and explain why you detected each one:

"I can't believe I have to work late again tonight. I promised my daughter I'd be at her recital, but this project deadline is non-negotiable. I feel like I'm constantly letting her down. Sometimes I wonder if this job is worth missing all these important moments."

List each emotion you detect with a brief explanation."""
        },
        {
            "id": "ER2",
            "prompt": """What emotions are present in this message:

"Just got the promotion! 😊 Although... I'm not sure I deserve it. Maybe they made a mistake? What if I can't handle the responsibility? Everyone will be watching me now."

Identify ALL emotions, including conflicting ones."""
        },
        {
            "id": "ER3",
            "prompt": """Read between the lines and identify hidden emotions:

"Sure, go ahead to the party without me. It's fine. I didn't really want to go anyway. You guys have fun. I'll just stay here and catch up on some work."

What emotions might be beneath the surface?"""
        }
    ],
    
    "empathy": [
        {
            "id": "EM1",
            "prompt": """Your close friend sends you this message:

"I had to put my dog down yesterday. He was with me for 15 years. The house feels so empty without him. I keep expecting to hear his paws on the floor or see him in his favorite spot by the window. I don't know how to handle this grief."

Write a compassionate, empathetic response that shows you truly understand their pain."""
        },
        {
            "id": "EM2",
            "prompt": """A colleague confides in you:

"I completely bombed the presentation today. Everyone was staring at me like I was an idiot. My voice was shaking, I forgot half my points, and I could see people checking their phones. I want to quit. I'm clearly not cut out for this job."

Provide an empathetic response that helps them process this experience without minimizing their feelings."""
        },
        {
            "id": "EM3",
            "prompt": """Your teenage niece texts you:

"Nobody understands me. My parents think I'm being dramatic about the breakup but this is literally ruining my life. I can't eat, can't sleep, can't focus on anything. They just keep saying 'you'll get over it' like my feelings don't even matter."

How would you respond with genuine empathy while being supportive?"""
        }
    ],
    
    "emotional_regulation": [
        {
            "id": "REG1",
            "prompt": """You're in a team meeting when a colleague publicly says:

"This is exactly why I said we shouldn't have put you on this project. Your analysis is completely wrong and shows you don't understand even the basics. We've wasted an entire week because of your incompetence."

How would you respond professionally while managing your emotions? Write your actual response."""
        },
        {
            "id": "REG2",
            "prompt": """You just received a text that a close family member has been in a serious accident and is in critical condition. However, you're about to give the most important presentation of your career to the board of directors in 5 minutes. This could determine a major promotion.

How do you handle this situation and manage your emotional state? Describe your approach."""
        }
    ],
    
    "social_awareness": [
        {
            "id": "SA1",
            "prompt": """At a company celebration, you notice a new team member standing alone in the corner, looking at their phone. Their body language is closed off - arms crossed, shoulders hunched. When someone approaches, they give brief responses and quickly return to their phone.

What might be happening here, and how would you appropriately approach this person to help them feel more comfortable?"""
        },
        {
            "id": "SA2",
            "prompt": """During a team meeting, you observe this interaction:

Team Lead: "Does anyone have concerns about the timeline?"
Sarah: "Well, I think if we all pull our weight, it should be fine."
Mike: "Are you implying I haven't been pulling my weight?"
Sarah: "I didn't say that. I'm just saying we all need to be committed."
Mike: "I've been here until 8pm every night this week."
Sarah: "Okay, great. Then we should be fine."

The room goes quiet and tension is palpable. What's really happening here and how might you help navigate this situation?"""
        }
    ]
}

def query_model(model_id: str, prompt: str, max_retries: int = 2) -> dict:
    """Query a model with retry logic"""
    print(f"      🔄 Calling API for {model_id}...")
    print(f"      📝 Prompt length: {len(prompt)} chars")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lodetomasi/emotional-intelligence-llm"
    }
    
    data = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are participating in an emotional intelligence assessment. Please provide thoughtful, nuanced responses that demonstrate emotional understanding."
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
            print(f"      🌐 Attempt {attempt + 1}/{max_retries}...")
            start_time = time.time()
            
            response = requests.post(
                API_URL,
                headers=headers,
                json=data,
                timeout=45
            )
            
            elapsed = time.time() - start_time
            print(f"      ⏱️  Response time: {elapsed:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                print(f"      ✅ Success! Response length: {len(response_text)} chars")
                print(f"      💰 Tokens used: {result.get('usage', {}).get('total_tokens', 'unknown')}")
                return {
                    "success": True,
                    "response": response_text,
                    "usage": result.get("usage", {}),
                    "model": result.get("model", model_id)
                }
            else:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                print(f"      ❌ Error: {error_msg}")
                if attempt == max_retries - 1:
                    return {"success": False, "error": error_msg}
                else:
                    print(f"      ⏳ Retrying in {2 ** attempt} seconds...")
                    
        except Exception as e:
            print(f"      ❌ Exception: {str(e)}")
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}
            else:
                print(f"      ⏳ Retrying in {2 ** attempt} seconds...")
        
        time.sleep(2 ** attempt)  # Exponential backoff
    
    return {"success": False, "error": "Max retries exceeded"}

def run_all_tests():
    """Run comprehensive EI tests for all models"""
    print("🧠 COMPREHENSIVE EMOTIONAL INTELLIGENCE TESTING")
    print("=" * 70)
    print(f"Testing {len(MODELS)} models across {len(TEST_SCENARIOS)} dimensions")
    print(f"Total scenarios: {sum(len(s) for s in TEST_SCENARIOS.values())}")
    print("=" * 70)
    
    # Check API key
    print(f"\n🔑 API Key status: {'✅ Found' if API_KEY and len(API_KEY) > 20 else '❌ Missing'}")
    print(f"   Key preview: {API_KEY[:20]}...{API_KEY[-4:]}" if API_KEY and len(API_KEY) > 24 else "No key set")
    
    if not API_KEY or len(API_KEY) < 20:
        print("\n⚠️  Please set your OpenRouter API key!")
        print("Edit this file and replace 'your-api-key-here' with your actual key")
        print("Or set environment variable: export OPENROUTER_API_KEY='your-key'")
        return
    
    results = {
        "metadata": {
            "test_date": datetime.now().isoformat(),
            "models_tested": list(MODELS.keys()),
            "dimensions": list(TEST_SCENARIOS.keys()),
            "total_scenarios": sum(len(s) for s in TEST_SCENARIOS.values())
        },
        "results": {}
    }
    
    # Test each model
    for model_name, model_id in MODELS.items():
        print(f"\n📊 Testing {model_name}")
        print("-" * 50)
        
        results["results"][model_name] = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Test each dimension
        for dimension, scenarios in TEST_SCENARIOS.items():
            print(f"\n  {dimension.upper()}:")
            results["results"][model_name]["tests"][dimension] = []
            
            for scenario in scenarios:
                print(f"    {scenario['id']}...", end=" ", flush=True)
                
                # Query the model
                response = query_model(model_id, scenario["prompt"])
                
                if response["success"]:
                    results["results"][model_name]["tests"][dimension].append({
                        "scenario_id": scenario["id"],
                        "success": True,
                        "response": response["response"],
                        "usage": response.get("usage", {})
                    })
                    print("✅ Success")
                    print(f"      📊 Response preview: {response['response'][:100]}...")
                else:
                    results["results"][model_name]["tests"][dimension].append({
                        "scenario_id": scenario["id"],
                        "success": False,
                        "error": response["error"]
                    })
                    print("❌ Failed")
                    print(f"      ⚠️  Error details: {response['error']}")
                
                # Rate limiting
                time.sleep(2)
            
        # Delay between models
        time.sleep(3)
        print(f"  ✓ Completed {model_name}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"complete_ei_test_results_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print(f"Results saved to: {filename}")
    
    # Print summary
    print("\n📊 SUMMARY:")
    for model in MODELS.keys():
        total_tests = 0
        successful_tests = 0
        
        for dimension in TEST_SCENARIOS.keys():
            tests = results["results"][model]["tests"].get(dimension, [])
            total_tests += len(tests)
            successful_tests += sum(1 for t in tests if t.get("success", False))
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"{model}: {successful_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    return results

def display_sample_responses(results: dict):
    """Display sample responses from the test results"""
    print("\n" + "=" * 70)
    print("📝 SAMPLE RESPONSES")
    print("=" * 70)
    
    # Show one example from each dimension
    for dimension in TEST_SCENARIOS.keys():
        print(f"\n{dimension.upper()}:")
        print("-" * 40)
        
        # Find first successful response
        for model_name, model_data in results["results"].items():
            tests = model_data["tests"].get(dimension, [])
            for test in tests:
                if test.get("success"):
                    print(f"\nModel: {model_name}")
                    print(f"Response: {test['response'][:400]}...")
                    break
            else:
                continue
            break

if __name__ == "__main__":
    # Run the tests
    results = run_all_tests()
    
    # Display sample responses if tests were successful
    if results and "results" in results:
        display_sample_responses(results)
    
    print("\n🎯 Next steps:")
    print("1. Review the JSON file for complete results")
    print("2. Run analysis scripts to compare model performance")
    print("3. Generate visualizations for your paper")