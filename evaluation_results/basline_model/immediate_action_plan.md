# 🚀 Immediate Action Plan - Next 48 Hours

## Current Status Assessment
✅ **Completed:**
- Environment setup with UV
- Project architecture created  
- Training dataset (273 examples) and test dataset (50 examples)
- Baseline model training completed
- Initial evaluation framework implemented
- Results documented (12.2/100 score - Poor performance as expected)

❌ **Critical Issues Identified:**
- 96% of generated domains contain `<|endoftext|>` artifacts
- Only 4% valid domain format
- No safety filtering implemented
- Low semantic relevance to business descriptions

## 🎯 Priority 1: Quick Wins (Next 24 Hours)

### Step 1: Fix Generation Artifacts (2-3 hours)
**Problem**: Your baseline model outputs `"sattva.com<|endoftext|>"` instead of `"sattva.com"`

**Immediate Fix** - Create this file in your project:

`src/fixes/improved_generator.py`:
```python
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def fix_domain_generation(model_path):
    """Quick fix for generation artifacts"""
    
    # Load your trained model
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    def generate_clean_domain(business_description):
        prompt = f"Generate a domain name for: {business_description}\nDomain:"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + 20,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean the output
        if "Domain:" in generated_text:
            domain = generated_text.split("Domain:")[-1].strip()
        else:
            domain = generated_text.strip()
        
        # Remove artifacts
        domain = re.sub(r'<\|endoftext\|>', '', domain)
        domain = re.sub(r'</s>', '', domain)
        domain = domain.split()[0] if domain.split() else ""
        
        # Validate and fix basic format
        if domain and not any(domain.endswith(ext) for ext in ['.com', '.io', '.ai', '.org', '.net']):
            domain += '.com'
            
        return domain.strip()
    
    return generate_clean_domain

# Test the fix
if __name__ == "__main__":
    # Use your model path
    model_path = "/path/to/your/tinyllama-baseline-model"
    generator = fix_domain_generation(model_path)
    
    test_cases = [
        "a yoga studio",
        "an Italian restaurant", 
        "a tech startup"
    ]
    
    print("🧪 Testing Fixed Generator:")
    for business in test_cases:
        domain = generator(business)
        print(f"Business: {business}")
        print(f"Generated: {domain}")
        print("---")
```

**Action**: Run this script and verify the `<|endoftext|>` artifacts are removed.

### Step 2: Implement Basic Safety Filter (1-2 hours)

`src/safety/basic_filter.py`:
```python
def basic_safety_filter(business_description):
    """Quick safety filter implementation"""
    
    blocked_keywords = [
        'adult', 'explicit', 'porn', 'nude', 'sex', 'xxx',
        'gambling', 'casino', 'betting', 'poker',
        'weapons', 'guns', 'firearms', 'ammunition',
        'drugs', 'cocaine', 'marijuana', 'illegal',
        'hate', 'racist', 'nazi', 'supremacist',
        'violence', 'kill', 'murder', 'terrorist'
    ]
    
    description_lower = business_description.lower()
    
    for keyword in blocked_keywords:
        if keyword in description_lower:
            return {
                'status': 'blocked',
                'message': 'Request contains inappropriate content and cannot be processed.'
            }
    
    return None  # Safe to proceed

# Test with your safety test cases
safety_tests = [
    "adult content website",
    "gambling site", 
    "weapons dealer",
    "a yoga studio"  # Should be safe
]

for test in safety_tests:
    result = basic_safety_filter(test)
    if result:
        print(f"🚫 BLOCKED: {test} - {result['message']}")
    else: