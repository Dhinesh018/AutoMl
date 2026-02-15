import json
from groq import Groq
from src.config import GROQ_API_KEY
from src.llm.llm_prompts import SYSTEM_PROMPT, build_dataset_prompt


def get_llm_decision(dataset_profile: dict, available_models: list) -> dict:
    """
    Use Groq LLM to intelligently select models based on dataset characteristics.
    
    Args:
        dataset_profile: Dataset statistics
        available_models: List of model configs
        
    Returns:
        dict with selected_models, reasoning, skipped_models
    """
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment!")
    
    client = Groq(api_key=GROQ_API_KEY)
    
    # Build prompt
    user_prompt = build_dataset_prompt(dataset_profile, available_models)
    
    try:
        # Call Groq API
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        response_text = completion.choices[0].message.content
        
        # Parse JSON response
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        llm_decision = json.loads(response_text)
        
        # Validate response structure
        if "selected_models" not in llm_decision:
            raise ValueError("LLM response missing 'selected_models'")
        
        # Map model names back to full configs
        selected_model_names = llm_decision["selected_models"]
        selected_configs = [
            model for model in available_models 
            if model["name"] in selected_model_names
        ]
        
        return {
            "selected_models": selected_configs,
            "reasoning": llm_decision.get("reasoning", "No reasoning provided"),
            "skipped_models": llm_decision.get("skipped_models", {}),
            "llm_raw_response": response_text
        }
        
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse LLM JSON response: {e}")
        print(f"Raw response: {response_text}")
        # Fallback to all models
        return {
            "selected_models": available_models,
            "reasoning": "LLM response parsing failed, using all models as fallback",
            "skipped_models": {},
            "llm_raw_response": response_text,
            "error": str(e)
        }
    
    except Exception as e:
        print(f"⚠️  LLM API call failed: {e}")
        # Fallback to all models
        return {
            "selected_models": available_models,
            "reasoning": f"LLM API failed: {str(e)}, using all models as fallback",
            "skipped_models": {},
            "error": str(e)
        }