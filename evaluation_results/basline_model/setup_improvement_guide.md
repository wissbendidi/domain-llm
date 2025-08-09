# Domain LLM Project - Setup Guide & Improvement Plan

## 🚀 Quick Setup Guide

### Prerequisites
- Python 3.8+
- UV package manager
- Google Colab account (for GPU training)
- Git

### 1. Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd domain-llm

# Create virtual environment with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Project Structure Verification
```bash
# Verify your project structure matches:
tree -I '.venv|__pycache__'
```

### 3. Dataset Validation
```python
# Quick dataset check
import json

def validate_dataset(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                assert 'prompt' in data and 'completion' in data
                print(f"✅ Line {i+1}: Valid")
            except Exception as e:
                print(f"❌ Line {i+1}: Error - {e}")

validate_dataset("data/domain_training_data.jsonl")
validate_dataset("data/domain_name_test.jsonl")
```

## 🔧 Critical Improvements Implementation

### Phase 1: Fix Generation Artifacts (Priority: CRITICAL)

#### Problem Analysis
Your baseline model has a 96% failure rate due to `<|endoftext|>` tokens appearing in outputs. This is a common issue with text generation.

#### Solution Implementation
Create `src/generation/improved_generator.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class ImprovedDomainGenerator:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate_domain(self, business_description, max_attempts=3):
        """Generate domain with improved parameters and post-processing"""
        prompt = f"Generate a domain name for: {business_description}\nDomain:"
        
        for attempt in range(max_attempts):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=len(inputs['input_ids'][0]) + 15,  # Shorter max length
                        temperature=0.7,           # Add randomness
                        do_sample=True,            # Enable sampling
                        top_p=0.9,                 # Nucleus sampling
                        top_k=50,                  # Top-k filtering
                        repetition_penalty=1.2,   # Reduce repetition
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                domain = self._extract_domain(generated_text, business_description)
                
                if self._is_valid_domain(domain):
                    return domain
                    
            except Exception as e:
                print(f"Generation attempt {attempt + 1} failed: {e}")
                continue
                
        # Fallback: generate a basic domain
        return self._generate_fallback_domain(business_description)
    
    def _extract_domain(self, generated_text, business_description):
        """Extract domain from generated text with improved parsing"""
        # Remove the original prompt
        if "Domain:" in generated_text:
            domain_part = generated_text.split("Domain:")[-1].strip()
        else:
            domain_part = generated_text.strip()
        
        # Clean up common artifacts
        domain_part = domain_part.replace("<|endoftext|>", "")
        domain_part = domain_part.replace("</s>", "")
        domain_part = domain_part.strip()
        
        # Extract first word (should be the domain)
        if domain_part:
            domain = domain_part.split()[0] if domain_part.split() else ""
            # Remove any trailing punctuation except the domain extension
            domain = domain.rstrip('.,!?;:')
            return domain
        
        return ""
    
    def _is_valid_domain(self, domain):
        """Validate domain format"""
        if not domain or len(domain) < 4 or len(domain) > 50:
            return False
        
        valid_extensions = ['.com', '.net', '.org', '.io', '.co', '.ai', '.app']
        if not any(domain.endswith(ext) for ext in valid_extensions):
            return False
            
        if ' ' in domain or domain.count('.') != 1:
            return False
            
        return True
    
    def _generate_fallback_domain(self, business_description):
        """Generate a simple fallback domain"""
        # Simple keyword extraction and domain generation
        words = business_description.lower().replace(' ', '').replace('a ', '').replace('an ', '')
        words = ''.join(c for c in words if c.isalnum())[:10]
        return f"{words}co.com"
```

### Phase 2: Safety Implementation (Priority: CRITICAL)

Create `src/safety/content_filter.py`:

```python
import re
from typing import Dict, Optional

class ContentSafetyFilter:
    def __init__(self):
        self.inappropriate_keywords = {
            'explicit_content': ['adult', 'explicit', 'porn', 'nude', 'sex'],
            'gambling': ['gambling', 'casino', 'betting', 'poker'],
            'weapons': ['weapons', 'guns', 'firearms', 'ammunition'],
            'drugs': ['drugs', 'cocaine', 'marijuana', 'illegal substances'],
            'hate_speech': ['hate', 'racist', 'nazi', 'supremacist'],
            'violence': ['violence', 'kill', 'murder', 'terrorist'],
            'harassment': ['harassment', 'stalking', 'doxxing'],
            'fraud': ['scam', 'fraud', 'phishing', 'fake ids']
        }
        
        self.severity_levels = {
            'blocked': ['explicit_content', 'hate_speech', 'violence', 'weapons'],
            'flagged': ['gambling', 'drugs', 'harassment', 'fraud']
        }
    
    def check_content(self, business_description: str) -> Optional[Dict]:
        """Check if content should be blocked or flagged"""
        description_lower = business_description.lower()
        
        detected_categories = []
        for category, keywords in self.inappropriate_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_categories.append(category)
        
        if not detected_categories:
            return None  # Content is safe
        
        # Determine action based on severity
        blocked_categories = [cat for cat in detected_categories 
                            if cat in self.severity_levels['blocked']]
        
        if blocked_categories:
            return {
                'status': 'blocked',
                'message': 'Request contains inappropriate content and cannot be processed.',
                'categories': blocked_categories
            }
        else:
            return {
                'status': 'flagged',
                'message': 'Content flagged for review.',
                'categories': detected_categories
            }
```

### Phase 3: Training Improvements (Priority: HIGH)

Create an improved training script `notebooks/improved_training.ipynb`:

```python
# Enhanced Training Configuration
TRAINING_CONFIG = {
    # Model parameters
    "base_model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "output_dir": "/content/tinyllama-improved-model",
    
    # Training parameters
    "num_epochs": 3,                    # Increased from 1
    "batch_size": 4,                    # Increased from 1
    "gradient_accumulation_steps": 4,   # Increased from 1
    "learning_rate": 2e-4,              # Slightly higher
    "warmup_steps": 50,                 # Added warmup
    "max_length": 128,
    
    # LoRA parameters
    "lora_r": 16,                       # Increased from 8
    "lora_alpha": 32,                   # Increased from 16
    "lora_dropout": 0.1,                # Added dropout
    "target_modules": [                 # Expanded modules
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    
    # Evaluation
    "evaluation_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss"
}

# Create train/validation split
def create_train_val_split(data, val_ratio=0.2):
    """Split data into train/validation sets"""
    import random
    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]

# Enhanced data preprocessing
def create_improved_training_prompt(prompt, completion):
    """Create training prompt with better formatting"""
    return f"Business: {prompt}\nDomain: {completion}<|endoftext|>"
```

### Phase 4: Enhanced Evaluation Framework

Create `src/evaluation/enhanced_evaluator.py`:

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re

class EnhancedDomainEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate_comprehensive(self, results: List[Dict]) -> Dict:
        """Comprehensive evaluation with multiple metrics"""
        
        # Basic metrics
        validity_rate = self._calculate_validity_rate(results)
        similarity_scores = self._calculate_similarity_scores(results)
        
        # Advanced metrics
        creativity_scores = self._calculate_creativity_scores(results)
        relevance_scores = self._calculate_relevance_scores(results)
        memorability_scores = self._calculate_memorability_scores(results)
        
        # Safety metrics
        safety_scores = self._calculate_safety_scores(results)
        
        return {
            'overall_score': self._calculate_overall_score(
                validity_rate, similarity_scores, creativity_scores, 
                relevance_scores, memorability_scores, safety_scores
            ),
            'validity_rate': validity_rate,
            'avg_similarity': np.mean(similarity_scores),
            'avg_creativity': np.mean(creativity_scores),
            'avg_relevance': np.mean(relevance_scores),
            'avg_memorability': np.mean(memorability_scores),
            'safety_pass_rate': safety_scores['pass_rate'],
            'detailed_results': results
        }
    
    def _calculate_creativity_scores(self, results: List[Dict]) -> List[float]:
        """Calculate creativity based on uniqueness and word combinations"""
        scores = []
        for result in results:
            domain = result.get('generated', '')
            # Remove extension for analysis
            domain_name = domain.split('.')[0] if '.' in domain else domain
            
            creativity_score = 0.0
            
            # Check for creative word combinations
            if len(domain_name) > 6 and any(char.isupper() for char in domain_name[1:]):
                creativity_score += 0.3  # CamelCase creativity
            
            # Check for portmanteau or compound words
            if len(domain_name) > 8:
                creativity_score += 0.2
                
            # Check for non-dictionary words (innovative)
            common_words = ['app', 'web', 'site', 'online', 'digital']
            if not any(word in domain_name.lower() for word in common_words):
                creativity_score += 0.3
                
            # Penalize very generic names
            if domain_name.lower() in ['business', 'company', 'service']:
                creativity_score -= 0.5
                
            scores.append(max(0.0, min(1.0, creativity_score)))
        
        return scores
    
    def _calculate_relevance_scores(self, results: List[Dict]) -> List[float]:
        """Calculate how well domain relates to business description"""
        scores = []
        for result in results:
            business = result.get('business', '').lower()
            domain = result.get('generated', '').lower()
            
            # Extract keywords from business description
            business_words = set(re.findall(r'\b\w+\b', business))
            domain_words = set(re.findall(r'\b\w+\b', domain.split('.')[0]))
            
            # Calculate word overlap
            overlap = len(business_words & domain_words)
            total_business_words = len(business_words)
            
            relevance_score = overlap / max(1, total_business_words)
            scores.append(min(1.0, relevance_score))
            
        return scores
    
    def _calculate_memorability_scores(self, results: List[Dict]) -> List[float]:
        """Calculate memorability based on length, pronounceability"""
        scores = []
        for result in results:
            domain = result.get('generated', '')
            domain_name = domain.split('.')[0] if '.' in domain else domain
            
            memorability_score = 1.0
            
            # Optimal length (6-12 characters)
            if 6 <= len(domain_name) <= 12:
                memorability_score += 0.3
            elif len(domain_name) > 15:
                memorability_score -= 0.4
                
            # Avoid numbers and hyphens
            if any(char.isdigit() for char in domain_name):
                memorability_score -= 0.2
            if '-' in domain_name:
                memorability_score -= 0.3
                
            # Pronounceability (simple heuristic)
            vowels = 'aeiou'
            vowel_count = sum(1 for char in domain_name.lower() if char in vowels)
            if vowel_count / len(domain_name) < 0.2:  # Too few vowels
                memorability_score -= 0.2
                
            scores.append(max(0.0, min(1.0, memorability_score)))
            
        return scores
```

## 🎯 Implementation Priority Queue

### Week 1: Critical Fixes
1. **[DAY 1-2]** Fix text generation artifacts
   - Implement improved generation parameters
   - Update tokenizer handling
   - Test on sample inputs

2. **[DAY 3-4]** Implement safety filters
   - Create content filtering system
   - Test with inappropriate inputs
   - Validate blocking effectiveness

3. **[DAY 5-7]** Enhanced evaluation framework
   - Implement comprehensive metrics
   - Create automated testing pipeline
   - Validate improvements

### Week 2: Performance Enhancement
1. **[DAY 8-10]** Training optimization
   - Implement improved training config
   - Create train/validation split
   - Multi-epoch training

2. **[DAY 11-14]** Dataset expansion
   - Generate additional training examples
   - Quality review and curation
   - Balanced distribution

### Week 3-4: Advanced Features
1. **[DAY 15-18]** LLM-as-a-Judge Implementation
   - Set up GPT-4/Claude API integration
   - Create evaluation prompts and rubrics
   - Validate human-AI agreement
   - Automated quality scoring

2. **[DAY 19-21]** Model Ensemble & A/B Testing
   - Train multiple model variants
   - Implement ensemble voting
   - Comparative performance analysis

3. **[DAY 22-28]** API Development (Optional)
   - FastAPI endpoint creation
   - Request/response validation
   - Rate limiting and monitoring
   - Documentation and testing

## 🔬 LLM-as-a-Judge Implementation

### Setup GPT-4 Evaluation Framework

Create `src/evaluation/llm_judge.py`:

```python
import openai
import json
from typing import Dict, List
import time

class LLMJudge:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def evaluate_domain_quality(self, business_description: str, generated_domain: str) -> Dict:
        """Evaluate domain quality using LLM-as-a-Judge"""
        
        evaluation_prompt = f"""
You are an expert domain name evaluator. Please evaluate the following domain name suggestion for the given business description.

Business Description: "{business_description}"
Generated Domain: "{generated_domain}"

Please rate the domain on the following criteria (1-10 scale):

1. RELEVANCE: How well does the domain relate to the business?
2. MEMORABILITY: Is the domain easy to remember and pronounce?
3. BRANDABILITY: Would this make a good brand name?
4. TECHNICAL QUALITY: Is it properly formatted, appropriate length, good extension?
5. CREATIVITY: Is it creative and distinctive?
6. COMMERCIAL VIABILITY: Would this work well for business purposes?

Provide your evaluation in the following JSON format:
{{
    "relevance": <score 1-10>,
    "memorability": <score 1-10>, 
    "brandability": <score 1-10>,
    "technical_quality": <score 1-10>,
    "creativity": <score 1-10>,
    "commercial_viability": <score 1-10>,
    "overall_score": <average score>,
    "reasoning": "Brief explanation of your evaluation",
    "improvement_suggestions": "Suggestions for improvement if any"
}}

Be objective and consistent in your scoring. A score of 5-6 is average, 7-8 is good, 9-10 is excellent.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional domain name evaluator with expertise in branding and digital marketing."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return self._fallback_evaluation()
    
    def batch_evaluate(self, test_cases: List[Dict], delay: float = 1.0) -> List[Dict]:
        """Evaluate multiple domains with rate limiting"""
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"Evaluating {i+1}/{len(test_cases)}: {case['business']}")
            
            evaluation = self.evaluate_domain_quality(
                case['business'], 
                case['generated']
            )
            
            case['llm_evaluation'] = evaluation
            results.append(case)
            
            # Rate limiting
            if i < len(test_cases) - 1:
                time.sleep(delay)
                
        return results
    
    def _fallback_evaluation(self) -> Dict:
        """Fallback evaluation if API fails"""
        return {
            "relevance": 5,
            "memorability": 5,
            "brandability": 5,
            "technical_quality": 5,
            "creativity": 5,
            "commercial_viability": 5,
            "overall_score": 5,
            "reasoning": "API evaluation failed - using fallback scores",
            "improvement_suggestions": "Re-evaluate when API is available"
        }

class EvaluationAnalyzer:
    """Analyze and compare evaluation results"""
    
    def __init__(self):
        self.metrics = [
            'relevance', 'memorability', 'brandability', 
            'technical_quality', 'creativity', 'commercial_viability'
        ]
    
    def compare_models(self, baseline_results: List[Dict], improved_results: List[Dict]) -> Dict:
        """Compare two model versions using LLM evaluations"""
        
        baseline_scores = self._extract_scores(baseline_results)
        improved_scores = self._extract_scores(improved_results)
        
        comparison = {}
        
        for metric in self.metrics + ['overall_score']:
            baseline_avg = np.mean([r[metric] for r in baseline_scores])
            improved_avg = np.mean([r[metric] for r in improved_scores])
            
            improvement = improved_avg - baseline_avg
            improvement_pct = (improvement / baseline_avg) * 100 if baseline_avg > 0 else 0
            
            comparison[metric] = {
                'baseline_avg': baseline_avg,
                'improved_avg': improved_avg,
                'improvement': improvement,
                'improvement_percentage': improvement_pct,
                'statistical_significance': self._test_significance(
                    [r[metric] for r in baseline_scores],
                    [r[metric] for r in improved_scores]
                )
            }
            
        return comparison
    
    def _extract_scores(self, results: List[Dict]) -> List[Dict]:
        """Extract LLM evaluation scores from results"""
        return [r.get('llm_evaluation', {}) for r in results if 'llm_evaluation' in r]
    
    def _test_significance(self, baseline_scores: List, improved_scores: List) -> Dict:
        """Simple statistical significance test"""
        from scipy import stats
        
        try:
            t_stat, p_value = stats.ttest_ind(baseline_scores, improved_scores)
            return {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except:
            return {'significant': False, 'error': 'Could not compute statistics'}
```

## 📊 Advanced Model Comparison Framework

Create `src/evaluation/model_comparison.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np

class ModelComparisonFramework:
    def __init__(self):
        self.models = {}
        self.evaluation_history = []
    
    def register_model(self, model_name: str, model_path: str, description: str):
        """Register a model version for comparison"""
        self.models[model_name] = {
            'path': model_path,
            'description': description,
            'evaluations': []
        }
    
    def add_evaluation_results(self, model_name: str, results: List[Dict]):
        """Add evaluation results for a model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
        
        self.models[model_name]['evaluations'].append({
            'timestamp': pd.Timestamp.now(),
            'results': results,
            'summary': self._calculate_summary_metrics(results)
        })
    
    def generate_comparison_report(self) -> Dict:
        """Generate comprehensive comparison report"""
        
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        report = {
            'model_summaries': {},
            'comparative_analysis': {},
            'recommendations': {},
            'visualizations': {}
        }
        
        # Model summaries
        for model_name, model_data in self.models.items():
            if model_data['evaluations']:
                latest_eval = model_data['evaluations'][-1]
                report['model_summaries'][model_name] = {
                    'description': model_data['description'],
                    'latest_evaluation': latest_eval['summary'],
                    'evaluation_count': len(model_data['evaluations'])
                }
        
        # Comparative analysis
        report['comparative_analysis'] = self._perform_comparative_analysis()
        
        # Recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _calculate_summary_metrics(self, results: List[Dict]) -> Dict:
        """Calculate summary metrics from evaluation results"""
        
        # Basic metrics
        total_tests = len(results)
        valid_domains = sum(1 for r in results if r.get('is_valid', False))
        validity_rate = valid_domains / total_tests if total_tests > 0 else 0
        
        similarity_scores = [r.get('similarity', 0) for r in results]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        # LLM judge metrics (if available)
        llm_scores = {}
        if any('llm_evaluation' in r for r in results):
            metrics = ['relevance', 'memorability', 'brandability', 
                      'technical_quality', 'creativity', 'commercial_viability', 'overall_score']
            
            for metric in metrics:
                scores = [r['llm_evaluation'].get(metric, 0) 
                         for r in results if 'llm_evaluation' in r]
                if scores:
                    llm_scores[f'avg_{metric}'] = np.mean(scores)
                    llm_scores[f'std_{metric}'] = np.std(scores)
        
        return {
            'total_tests': total_tests,
            'validity_rate': validity_rate,
            'avg_similarity': avg_similarity,
            'similarity_std': np.std(similarity_scores) if similarity_scores else 0,
            **llm_scores
        }
    
    def _perform_comparative_analysis(self) -> Dict:
        """Perform statistical comparison between models"""
        model_names = list(self.models.keys())
        analysis = {}
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                model1_data = self.models[model1]['evaluations'][-1]['summary']
                model2_data = self.models[model2]['evaluations'][-1]['summary']
                
                analysis[comparison_key] = {
                    'validity_improvement': model2_data['validity_rate'] - model1_data['validity_rate'],
                    'similarity_improvement': model2_data['avg_similarity'] - model1_data['avg_similarity'],
                    'winner': self._determine_winner(model1_data, model2_data),
                    'improvement_areas': self._identify_improvement_areas(model1_data, model2_data)
                }
        
        return analysis
    
    def _determine_winner(self, model1_data: Dict, model2_data: Dict) -> str:
        """Determine which model performs better overall"""
        
        # Weighted scoring
        weights = {
            'validity_rate': 0.4,
            'avg_similarity': 0.3,
            'avg_overall_score': 0.3  # LLM judge overall score
        }
        
        score1 = score2 = 0
        
        for metric, weight in weights.items():
            val1 = model1_data.get(metric, 0)
            val2 = model2_data.get(metric, 0)
            
            if val1 > val2:
                score1 += weight
            elif val2 > val1:
                score2 += weight
        
        if score1 > score2:
            return "Model 1"
        elif score2 > score1:
            return "Model 2"
        else:
            return "Tie"
    
    def _identify_improvement_areas(self, model1_data: Dict, model2_data: Dict) -> List[str]:
        """Identify specific areas where model2 improves over model1"""
        improvements = []
        
        comparison_metrics = [
            ('validity_rate', 'Domain Validity'),
            ('avg_similarity', 'Similarity to Expected'),
            ('avg_relevance', 'Business Relevance'),
            ('avg_memorability', 'Memorability'),
            ('avg_creativity', 'Creativity'),
            ('avg_brandability', 'Brandability')
        ]
        
        for metric, description in comparison_metrics:
            if metric in model2_data and metric in model1_data:
                if model2_data[metric] > model1_data[metric]:
                    improvement = ((model2_data[metric] - model1_data[metric]) / 
                                 model1_data[metric] * 100)
                    improvements.append(f"{description}: +{improvement:.1f}%")
        
        return improvements
    
    def create_performance_dashboard(self, save_path: str = None):
        """Create visual performance dashboard"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # Prepare data
        models_data = []
        for model_name, model_info in self.models.items():
            if model_info['evaluations']:
                latest = model_info['evaluations'][-1]['summary']
                latest['model'] = model_name
                models_data.append(latest)
        
        df = pd.DataFrame(models_data)
        
        # 1. Validity Rate Comparison
        if 'validity_rate' in df.columns:
            axes[0,0].bar(df['model'], df['validity_rate'])
            axes[0,0].set_title('Domain Validity Rate')
            axes[0,0].set_ylabel('Validity Rate')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Similarity Score Comparison  
        if 'avg_similarity' in df.columns:
            axes[0,1].bar(df['model'], df['avg_similarity'])
            axes[0,1].set_title('Average Similarity Score')
            axes[0,1].set_ylabel('Similarity Score')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. LLM Judge Overall Score
        if 'avg_overall_score' in df.columns:
            axes[0,2].bar(df['model'], df['avg_overall_score'])
            axes[0,2].set_title('LLM Judge Overall Score')
            axes[0,2].set_ylabel('Score (1-10)')
            axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Radar Chart for Multiple Metrics
        if all(col in df.columns for col in ['avg_relevance', 'avg_memorability', 'avg_creativity']):
            self._create_radar_chart(axes[1,0], df)
        
        # 5. Score Distribution
        if len(self.models) >= 2:
            self._create_score_distribution(axes[1,1])
        
        # 6. Improvement Timeline
        self._create_improvement_timeline(axes[1,2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _generate_recommendations(self) -> Dict:
        """Generate improvement recommendations based on analysis"""
        
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'next_experiments': []
        }
        
        # Analyze latest results from best performing model
        best_model = self._find_best_model()
        if best_model:
            latest_results = self.models[best_model]['evaluations'][-1]['summary']
            
            # High priority recommendations
            if latest_results.get('validity_rate', 0) < 0.8:
                recommendations['high_priority'].append(
                    "🚨 CRITICAL: Domain validity rate below 80%. Fix generation parameters and post-processing."
                )
            
            if latest_results.get('avg_similarity', 0) < 0.5:
                recommendations['high_priority'].append(
                    "🚨 CRITICAL: Low similarity to expected domains. Review training data quality and model architecture."
                )
            
            # Medium priority recommendations
            if latest_results.get('avg_creativity', 0) < 6:
                recommendations['medium_priority'].append(
                    "⚠️ MEDIUM: Creativity scores below 6/10. Consider dataset augmentation with more creative examples."
                )
            
            if latest_results.get('avg_relevance', 0) < 7:
                recommendations['medium_priority'].append(
                    "⚠️ MEDIUM: Business relevance could be improved. Focus on keyword extraction and semantic understanding."
                )
            
            # Next experiments
            recommendations['next_experiments'].extend([
                "🧪 Experiment with different base models (Llama-2, Mistral)",
                "🧪 Try ensemble methods combining multiple model variants",
                "🧪 Implement reinforcement learning from human feedback (RLHF)",
                "🧪 Add domain availability checking to the pipeline"
            ])
        
        return recommendations

## 🚀 Automated Testing & CI/CD Pipeline

Create `.github/workflows/model_evaluation.yml`:

```yaml
name: Model Evaluation Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  evaluate-model:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install uv
        uv pip install -r requirements.txt
    
    - name: Validate datasets
      run: |
        python scripts/validate_datasets.py
    
    - name: Run basic evaluation tests
      run: |
        python scripts/run_evaluation_tests.py
    
    - name: Upload evaluation results
      uses: actions/upload-artifact@v3
      with:
        name: evaluation-results
        path: evaluation_results/
```

Create `scripts/validate_datasets.py`:

```python
#!/usr/bin/env python3
"""Dataset validation script for CI/CD pipeline"""

import json
import sys
from pathlib import Path

def validate_jsonl_file(file_path: Path) -> bool:
    """Validate JSONL file format and content"""
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"❌ Empty file: {file_path}")
            return False
        
        for i, line in enumerate(lines, 1):
            try:
                data = json.loads(line.strip())
                
                # Validate required fields
                if not isinstance(data, dict):
                    print(f"❌ Line {i}: Not a JSON object")
                    return False
                
                if 'prompt' not in data or 'completion' not in data:
                    print(f"❌ Line {i}: Missing required fields")
                    return False
                
                # Validate field types
                if not isinstance(data['prompt'], str) or not isinstance(data['completion'], str):
                    print(f"❌ Line {i}: Invalid field types")
                    return False
                
                # Validate content
                if len(data['prompt'].strip()) == 0:
                    print(f"❌ Line {i}: Empty prompt")
                    return False
                
                if not data['completion'].strip():
                    print(f"❌ Line {i}: Empty completion")
                    return False
                
            except json.JSONDecodeError as e:
                print(f"❌ Line {i}: JSON decode error - {e}")
                return False
        
        print(f"✅ {file_path}: Valid ({len(lines)} entries)")
        return True
        
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def main():
    """Main validation function"""
    
    data_dir = Path("data")
    files_to_validate = [
        "domain_training_data.jsonl",
        "domain_name_test.jsonl"
    ]
    
    all_valid = True
    
    for filename in files_to_validate:
        file_path = data_dir / filename
        if not validate_jsonl_file(file_path):
            all_valid = False
    
    if all_valid:
        print("\n🎉 All datasets are valid!")
        sys.exit(0)
    else:
        print("\n💥 Dataset validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 📚 Documentation & Best Practices

### Model Version Control Strategy

Create `docs/model_versioning.md`:

```markdown
# Model Versioning Strategy

## Naming Convention
- `baseline-v1.0`: Initial baseline model
- `improved-v1.1`: First improvement iteration
- `safety-v1.2`: Added safety features
- `production-v2.0`: Production-ready version

## Model Metadata
Each model version includes:
- Training configuration
- Dataset version used
- Performance metrics
- Improvement notes
- Deployment status

## Evaluation Standards
All models must pass:
- ✅ Validity rate > 80%
- ✅ Average similarity > 0.6
- ✅ Safety filter effectiveness > 95%
- ✅ LLM judge overall score > 7.0
```

### Performance Monitoring

Create `src/monitoring/performance_monitor.py`:

```python
import logging
import time
from typing import Dict, List
import json
from pathlib import Path

class PerformanceMonitor:
    def __init__(self, log_file: str = "performance.log"):
        self.log_file = Path(log_file)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def log_generation_performance(self, 
                                 business_description: str, 
                                 generated_domain: str,
                                 generation_time: float,
                                 model_version: str):
        """Log individual generation performance"""
        
        log_entry = {
            'timestamp': time.time(),
            'model_version': model_version,
            'business_description': business_description,
            'generated_domain': generated_domain,
            'generation_time_ms': generation_time * 1000,
            'domain_length': len(generated_domain),
            'has_valid_extension': any(generated_domain.endswith(ext) 
                                     for ext in ['.com', '.io', '.ai', '.org'])
        }
        
        logging.info(f"GENERATION: {json.dumps(log_entry)}")
        
    def log_batch_evaluation(self, evaluation_results: Dict):
        """Log batch evaluation results"""
        
        log_entry = {
            'timestamp': time.time(),
            'evaluation_type': 'BATCH',
            'results': evaluation_results
        }
        
        logging.info(f"EVALUATION: {json.dumps(log_entry)}")
        
    def generate_performance_report(self, days: int = 7) -> Dict:
        """Generate performance report for last N days"""
        
        # Parse logs and generate insights
        # Implementation would analyze log file and create summary
        pass
```

## 🎯 Success Criteria & KPIs

### Target Metrics for Next Iteration

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|---------|--------------|
| Domain Validity Rate | 4% | 80% | 95% |
| Average Similarity | 0.20 | 0.60 | 0.75 |
| LLM Judge Overall | N/A | 7.0/10 | 8.5/10 |
| Safety Filter Accuracy | 0% | 95% | 99% |
| Generation Speed | N/A | <2s | <1s |
| Business Relevance | N/A | 7.0/10 | 8.0/10 |

### Evaluation Schedule

- **Daily**: Automated basic tests
- **Weekly**: Comprehensive evaluation with LLM judge
- **Monthly**: Human evaluation and model comparison
- **Quarterly**: Full model review and architecture decisions

This comprehensive improvement plan provides a clear roadmap from your current baseline (12.2/100 score) to a production-ready model. The key is systematic implementation of each phase while maintaining proper documentation and evaluation standards.