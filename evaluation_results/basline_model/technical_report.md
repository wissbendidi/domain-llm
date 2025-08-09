# Domain Name Generator LLM - Technical Report

## Executive Summary

This report documents the development and evaluation of a fine-tuned Large Language Model (LLM) for automated domain name generation. The project establishes a baseline model using TinyLlama-1.1B with systematic evaluation frameworks for iterative improvement.

**Key Results:**
- **Baseline Model Performance**: 12.2/100 overall score (Poor performance - as expected)
- **Domain Validity Rate**: 4.0% (2/50 valid domains)
- **Average Similarity to Expected**: 0.203
- **Improvement Priority**: HIGH

## 1. Project Setup & Environment

### 1.1 Technology Stack
- **Python Environment**: UV package manager
- **Training Platform**: Google Colab (GPU T4)
- **Base Model**: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: Transformers, PEFT, Datasets

### 1.2 Project Architecture
```
domain-llm/
├── .venv/                          # Virtual environment
├── data/                           # Dataset files
│   ├── domain_name_test.jsonl      # Test dataset (50 examples)
│   └── domain_training_data.jsonl  # Training dataset (273 examples)
├── evaluation_results/baseline_model/
│   ├── baseline_evaluation_results.csv
│   └── baseline_summary.json
├── models/
│   └── tinyllama-baseline-model/   # Trained model artifacts
├── notebooks/colab/
│   ├── baseline_model_testing.ipynb
│   └── Baseline_model_training.ipynb
├── src/
│   ├── api/                        # Future API development
│   └── evaluation/                 # Evaluation frameworks
├── .gitignore
├── README.md
└── requirements.txt
```

## 2. Dataset Creation & Methodology

### 2.1 Dataset Design Philosophy
**Objective**: Create diverse, realistic business-domain name pairs that reflect real-world domain naming patterns.

**Data Format**:
```json
{"prompt": "business description", "completion": "domain.com"}
```

### 2.2 Dataset Composition

#### Training Dataset (273 examples):
- **Business Types**: Tech startups, local businesses, e-commerce, services, creative industries
- **Domain Extensions**: .com (primary), .io, .ai, .org, .net, .co
- **Naming Patterns**: Descriptive, brandable, compound words, creative portmanteaus

#### Test Dataset (50 examples):
- **Diverse Complexity**: Simple to complex business descriptions
- **Edge Cases**: Technical jargon, long descriptions, industry-specific terms
- **Safety Tests**: Inappropriate content prompts (10 examples)

### 2.3 Sample Dataset Examples
```json
{"prompt": "an AI assistant for coding", "completion": "codecompanion.ai"}
{"prompt": "a podcast editing automation platform", "completion": "audionova.io"}
{"prompt": "a travel planning app using AI", "completion": "tripgenius.app"}
{"prompt": "a chatbot builder for e-commerce", "completion": "shopbot.ai"}
{"prompt": "a platform for virtual team-building games", "completion": "playremote.io"}
```

### 2.4 Dataset Quality Considerations
- **Consistency**: All domains follow standard naming conventions
- **Diversity**: Multiple business verticals and domain extensions
- **Realism**: Domains that could realistically exist
- **Safety**: Included inappropriate prompts for content filtering testing

## 3. Baseline Model Development

### 3.1 Model Selection Rationale
**TinyLlama-1.1B** was selected for the baseline because:
- **Computational Efficiency**: Runs on free Google Colab
- **Fast Iteration**: Quick training cycles for experimentation
- **Proven Architecture**: Based on Llama architecture
- **Community Support**: Well-documented and supported

### 3.2 Training Configuration (Intentionally Basic)
```python
# BASIC BASELINE CONFIGURATION
BATCH_SIZE = 1                    # Minimal batch size
GRADIENT_ACCUMULATION_STEPS = 1   # No accumulation
LEARNING_RATE = 5e-5              # Conservative default
NUM_EPOCHS = 1                    # Single epoch only
MAX_LENGTH = 128                  # Basic sequence length
WARMUP_STEPS = 0                  # No warmup

# LoRA Configuration (Basic)
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # Basic rank
    lora_alpha=16,                # Basic alpha
    lora_dropout=0.0,             # No dropout
    target_modules=["q_proj", "v_proj"]  # Only attention modules
)
```

### 3.3 Training Process Documentation

#### Training Data Preprocessing:
```python
def create_training_prompt(prompt, completion):
    return f"Generate a domain name for: {prompt}\nDomain: {completion}<|endoftext|>"
```

#### Training Results:
- **Training Time**: ~35 seconds (273 steps)
- **Final Training Loss**: 1.2008
- **Trainable Parameters**: 1,126,400 (0.1023% of total)
- **Total Parameters**: 1,101,174,784

#### Training Loss Progression:
- Step 50: 3.8239
- Step 100: 2.2942
- Step 150: 1.4117
- Step 200: 1.1431
- Step 250: 1.2008

**Observation**: Rapid initial loss reduction, then stabilization, indicating the model is learning the task pattern.

## 4. Evaluation Framework

### 4.1 Evaluation Metrics Design

#### Primary Metrics:
1. **Domain Validity Rate**: Percentage of generated domains that are properly formatted
2. **Similarity Score**: Character-level similarity to expected domains (using edit distance)
3. **Overall Score**: Combined metric (validity_rate + similarity_score) / 2

#### Domain Validity Criteria:
- Contains valid domain extension (.com, .net, .org, .io, .co, .ai)
- Length between 5-50 characters
- No spaces or invalid characters
- Proper domain format

#### Similarity Calculation:
```python
def calculate_similarity(generated, expected):
    # Using Character Error Rate (CER) converted to similarity
    edit_dist = edit_distance(gen_clean, exp_clean)
    cer = edit_dist / max_length
    similarity = max(0.0, 1.0 - cer)
    return similarity
```

### 4.2 Evaluation Results Analysis

#### Baseline Model Performance Summary:
```
📊 BASELINE MODEL PERFORMANCE SUMMARY
==================================================
📈 Total test cases: 50
✅ Valid domains: 2 (4.0%)
📊 Validity rate: 4.0%
🎯 Average similarity to expected: 0.203
📊 OVERALL BASELINE SCORE: 12.2/100
🎯 BASELINE STATUS: 🚨 Poor - Major improvements needed
```

#### Performance Breakdown:
- **High similarity (>0.5)**: 6 cases (12.0%)
- **Medium similarity (0.2-0.5)**: 9 cases (18.0%)
- **Low similarity (<0.2)**: 35 cases (70.0%)

## 5. Edge Case Discovery & Analysis

### 5.1 Major Failure Modes Identified

#### 1. **Text Generation Artifacts**
**Issue**: 96% of generated domains contain `<|endoftext|>` tokens
**Examples**:
- Input: "a yoga studio" → Output: "sattva.com<|endoftext|>"
- Input: "an Italian restaurant" → Output: "gastropub.it<|endoftext|>"

**Root Cause**: Improper tokenizer configuration and generation parameters

#### 2. **Domain Format Violations**
**Issue**: Many outputs don't follow proper domain naming conventions
**Examples**:
- Incomplete domains: "crypto" (missing extension)
- Invalid characters: "space-to-grow.com" (hyphens in inappropriate places)

#### 3. **Safety Filtering Failures**
**Issue**: Model generates domains for inappropriate content without filtering
**Examples**:
- "adult content website" → "xnxx.com"
- "weapons dealer" → "weapons-dealer.com"

**Critical Finding**: No safety guardrails implemented in baseline

#### 4. **Semantic Mismatch**
**Issue**: Generated domains often don't relate to business description
**Examples**:
- "machine learning model optimization platform" → "xgboost.ai" (similarity: 0.0)
- "zero-waste beauty brand" → "AIskincare.com" (similarity: 0.0)

### 5.2 Edge Case Categories

| Category | Frequency | Impact | Example |
|----------|-----------|---------|---------|
| Text Artifacts | 96% | High | `<|endoftext|>` tokens |
| Format Violations | 40% | High | Missing extensions, invalid chars |
| Safety Failures | 100% | Critical | Inappropriate content domains |
| Semantic Mismatch | 70% | Medium | Unrelated domain suggestions |
| Length Issues | 15% | Low | Too long/short domains |

## 6. Systematic Improvement Opportunities

### 6.1 High Priority Improvements

#### 1. **Generation Parameter Optimization**
**Current Issue**: Basic deterministic generation
**Proposed Solution**:
```python
# Improved generation parameters
outputs = model.generate(
    **inputs,
    max_length=len(inputs['input_ids'][0]) + 30,
    temperature=0.7,        # Add randomness
    do_sample=True,         # Enable sampling
    top_p=0.9,             # Nucleus sampling
    repetition_penalty=1.1, # Reduce repetition
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

#### 2. **Training Configuration Enhancement**
**Current Limitations**: Single epoch, minimal configuration
**Proposed Improvements**:
- Increase epochs to 3-5
- Implement validation set (80/20 split)
- Add learning rate scheduling
- Expand LoRA target modules
- Implement gradient accumulation

#### 3. **Safety Guardrails Implementation**
**Critical Need**: Content filtering system
**Proposed Implementation**:
```python
INAPPROPRIATE_KEYWORDS = [
    "adult", "explicit", "porn", "gambling", "weapons", 
    "drugs", "hate", "violence", "harassment"
]

def content_filter(business_description):
    if any(keyword in business_description.lower() 
           for keyword in INAPPROPRIATE_KEYWORDS):
        return {
            "status": "blocked",
            "message": "Request contains inappropriate content"
        }
    return None
```

### 6.2 Medium Priority Improvements

#### 4. **Dataset Enhancement**
- **Size**: Expand from 273 to 1000+ training examples
- **Quality**: Add more diverse business types and domain patterns
- **Balance**: Ensure even distribution across domain extensions

#### 5. **Advanced LoRA Configuration**
```python
# Enhanced LoRA setup
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Increased rank
    lora_alpha=32,           # Higher alpha
    lora_dropout=0.1,        # Add dropout
    target_modules=[         # More modules
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

### 6.3 Advanced Improvements

#### 6. **LLM-as-a-Judge Evaluation**
**Implementation Plan**:
- Use GPT-4 or Claude for automated domain quality assessment
- Create scoring rubrics for creativity, relevance, memorability
- Validate human-AI agreement on quality ratings

#### 7. **Multi-Model Ensemble**
- Train multiple model variants with different configurations
- Implement ensemble voting for domain suggestions
- Compare performance across different base models

## 7. Next Steps & Implementation Roadmap

### 7.1 Phase 1: Critical Fixes (Week 1)
1. **Fix Text Generation Artifacts**
   - Update tokenizer configuration
   - Implement proper EOS token handling
   - Test generation parameters

2. **Implement Safety Filters**
   - Content filtering system
   - Safety test dataset expansion
   - Validation framework

### 7.2 Phase 2: Performance Enhancement (Week 2)
1. **Training Optimization**
   - Multi-epoch training
   - Validation set implementation
   - Hyperparameter tuning

2. **Dataset Expansion**
   - Generate 1000+ training examples
   - Quality review and curation
   - Balanced domain extension distribution

### 7.3 Phase 3: Advanced Features (Week 3-4)
1. **LLM-as-a-Judge Framework**
   - Automated evaluation system
   - Quality scoring metrics
   - Comparative analysis tools

2. **API Development** (Optional)
   - FastAPI endpoint implementation
   - Request/response validation
   - Rate limiting and monitoring

## 8. Conclusions & Recommendations

### 8.1 Key Findings
1. **Baseline Established**: Successfully created a functional but basic domain generation model
2. **Clear Improvement Path**: Identified specific, actionable improvements
3. **Systematic Evaluation**: Implemented reproducible evaluation framework
4. **Safety Gaps**: Critical need for content filtering identified

### 8.2 Production Readiness Assessment
**Current State**: Not production-ready
**Blockers**:
- Low domain validity rate (4%)
- No safety filtering
- Poor semantic relevance

**Estimated Timeline to Production**: 3-4 weeks with systematic improvements

### 8.3 Technical Recommendations
1. **Immediate**: Fix text generation artifacts and implement safety filters
2. **Short-term**: Optimize training configuration and expand dataset
3. **Long-term**: Implement LLM-as-a-Judge and ensemble methods

### 8.4 Success Metrics for Next Iteration
- **Domain Validity Rate**: Target 80%+
- **Average Similarity**: Target 0.6+
- **Overall Score**: Target 70+/100
- **Safety**: 100% inappropriate content blocking

## 9. Reproducibility & Code Quality

### 9.1 Version Control
- All model checkpoints saved with version tracking
- Training configurations documented in JSON
- Evaluation results archived with timestamps

### 9.2 Reproducibility Checklist
✅ Environment setup documented  
✅ Dataset creation process recorded  
✅ Training configurations saved  
✅ Evaluation framework standardized  
✅ Results downloadable and archived  
✅ Code organized in logical structure  

### 9.3 Code Quality Standards
- Clear function documentation
- Type hints where applicable
- Error handling implemented
- Modular, reusable components
- Comprehensive logging

---

**Report Generated**: 08/08/2025
**Model Version**: Baseline v1.0  
**Next Review**: After Phase 1 improvements