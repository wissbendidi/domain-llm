# src/evaluation/llm_judge.py

import torch
import json
import time
import logging
from typing import Dict, List, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline
)
import re
from dataclasses import dataclass

@dataclass
class JudgeResult:
    """Structured result from LLM judge evaluation"""
    relevance: float
    memorability: float
    brandability: float
    technical_quality: float
    creativity: float
    commercial_viability: float
    overall_score: float
    reasoning: str
    improvement_suggestions: str
    confidence: float = 0.0

class FreeLLMJudge:
    """
    LLM-as-a-Judge implementation using free open-source models
    
    Supported models:
    - microsoft/DialoGPT-medium (lightweight, fast)
    - microsoft/DialoGPT-large (better quality)
    - HuggingFaceH4/zephyr-7b-beta (high quality, larger)
    - mistralai/Mistral-7B-Instruct-v0.1 (excellent quality)
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "auto",
                 cache_dir: str = "./models/judge_cache"):
        """
        Initialize the LLM Judge
        
        Args:
            model_name: Name of the model to use for judging
            device: Device to run on ('cpu', 'cuda', or 'auto')
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self._initialize_model()
        
        # Evaluation templates
        self.evaluation_template = self._create_evaluation_template()
        
    def _initialize_model(self):
        """Initialize the model and tokenizer"""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # For better performance, use different approaches based on model
            if "DialoGPT" in self.model_name:
                # Use pipeline for DialoGPT (optimized for conversation)
                self.generator = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.tokenizer = self.generator.tokenizer
                
            else:
                # Use direct model loading for instruction-tuned models
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    cache_dir=self.cache_dir
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                self.generator = None
                
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Fallback to a smaller model
            self.logger.info("Falling back to microsoft/DialoGPT-medium")
            self.model_name = "microsoft/DialoGPT-medium"
            self._initialize_model()
    
    def _create_evaluation_template(self) -> str:
        """Create the evaluation prompt template"""
        return """You are an expert domain name evaluator. Evaluate the following domain name for the given business.

Business Description: "{business_description}"
Generated Domain: "{generated_domain}"

Rate each criterion from 1-10 (1=very poor, 5=average, 10=excellent):

1. RELEVANCE: How well does the domain relate to the business?
2. MEMORABILITY: Is it easy to remember and pronounce?
3. BRANDABILITY: Would this make a good brand name?
4. TECHNICAL_QUALITY: Proper format, good length, appropriate extension?
5. CREATIVITY: Is it creative and distinctive?
6. COMMERCIAL_VIABILITY: Would this work for business purposes?

Respond ONLY with this JSON format:
{{
    "relevance": <score>,
    "memorability": <score>,
    "brandability": <score>,
    "technical_quality": <score>,
    "creativity": <score>,
    "commercial_viability": <score>,
    "overall_score": <average_score>,
    "reasoning": "<brief explanation>",
    "improvement_suggestions": "<suggestions if any>"
}}

Evaluation:"""

    def evaluate_domain(self, 
                       business_description: str, 
                       generated_domain: str,
                       max_retries: int = 3) -> JudgeResult:
        """
        Evaluate a single domain name
        
        Args:
            business_description: The business description prompt
            generated_domain: The generated domain to evaluate
            max_retries: Number of retries if parsing fails
            
        Returns:
            JudgeResult object with evaluation scores
        """
        
        # Clean inputs
        business_description = business_description.strip()
        generated_domain = generated_domain.strip()
        
        # Create evaluation prompt
        prompt = self.evaluation_template.format(
            business_description=business_description,
            generated_domain=generated_domain
        )
        
        for attempt in range(max_retries):
            try:
                # Generate evaluation
                response = self._generate_response(prompt)
                
                # Parse JSON response
                result = self._parse_evaluation_response(response)
                
                if result:
                    # Add confidence score based on response quality
                    result.confidence = self._calculate_confidence(response)
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Evaluation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error("All evaluation attempts failed, using fallback")
                    return self._create_fallback_result(business_description, generated_domain)
        
        return self._create_fallback_result(business_description, generated_domain)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from the model"""
        
        if self.generator:
            # Use pipeline
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + 200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            return response[0]['generated_text'][len(prompt):].strip()
            
        else:
            # Use direct model generation
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 200,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
    
    def _parse_evaluation_response(self, response: str) -> Optional[JudgeResult]:
        """Parse the JSON response from the model"""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if not json_match:
                return None
                
            json_str = json_match.group()
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'relevance', 'memorability', 'brandability', 
                'technical_quality', 'creativity', 'commercial_viability'
            ]
            
            for field in required_fields:
                if field not in data:
                    return None
                    
                # Ensure scores are in valid range
                score = float(data[field])
                if not (1 <= score <= 10):
                    data[field] = max(1, min(10, score))
            
            # Calculate overall score if not provided
            if 'overall_score' not in data:
                scores = [data[field] for field in required_fields]
                data['overall_score'] = sum(scores) / len(scores)
            
            # Ensure text fields exist
            data['reasoning'] = data.get('reasoning', 'No reasoning provided')
            data['improvement_suggestions'] = data.get('improvement_suggestions', 'No suggestions provided')
            
            return JudgeResult(
                relevance=data['relevance'],
                memorability=data['memorability'],
                brandability=data['brandability'],
                technical_quality=data['technical_quality'],
                creativity=data['creativity'],
                commercial_viability=data['commercial_viability'],
                overall_score=data['overall_score'],
                reasoning=data['reasoning'],
                improvement_suggestions=data['improvement_suggestions']
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse evaluation response: {e}")
            return None
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response quality"""
        
        confidence = 1.0
        
        # Check for JSON format
        if '{' not in response or '}' not in response:
            confidence *= 0.5
            
        # Check for reasoning length
        if len(response) < 50:
            confidence *= 0.7
            
        # Check for improvement suggestions
        if 'improvement' not in response.lower():
            confidence *= 0.8
            
        return max(0.1, min(1.0, confidence))
    
    def _create_fallback_result(self, business_description: str, generated_domain: str) -> JudgeResult:
        """Create fallback evaluation when model fails"""
        
        # Simple rule-based fallback evaluation
        domain_clean = generated_domain.lower().replace('.com', '').replace('.io', '').replace('.ai', '')
        business_clean = business_description.lower()
        
        # Basic relevance check
        relevance = 7.0 if any(word in domain_clean for word in business_clean.split()[:3]) else 4.0
        
        # Basic technical quality check
        technical_quality = 8.0 if len(domain_clean) <= 15 and '.' in generated_domain else 5.0
        
        # Basic memorability (shorter is better)
        memorability = max(3.0, 10.0 - len(domain_clean) * 0.3)
        
        return JudgeResult(
            relevance=relevance,
            memorability=memorability,
            brandability=6.0,  # Neutral
            technical_quality=technical_quality,
            creativity=5.0,    # Neutral
            commercial_viability=6.0,  # Neutral
            overall_score=(relevance + memorability + technical_quality + 17.0) / 6.0,
            reasoning="Fallback evaluation due to model response parsing failure",
            improvement_suggestions="Re-evaluate with working model for detailed feedback",
            confidence=0.3
        )
    
    def batch_evaluate(self, 
                      test_cases: List[Dict[str, str]], 
                      delay: float = 0.5,
                      progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Evaluate multiple domain names in batch
        
        Args:
            test_cases: List of dicts with 'business' and 'generated' keys
            delay: Delay between evaluations to prevent overloading
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of test cases with added 'llm_evaluation' field
        """
        
        results = []
        total = len(test_cases)
        
        self.logger.info(f"Starting batch evaluation of {total} cases")
        
        for i, case in enumerate(test_cases):
            try:
                if progress_callback:
                    progress_callback(i, total)
                
                business = case.get('business', case.get('prompt', ''))
                generated = case.get('generated', case.get('completion', ''))
                
                # Perform evaluation
                evaluation = self.evaluate_domain(business, generated)
                
                # Add evaluation to case
                case_with_eval = case.copy()
                case_with_eval['llm_evaluation'] = {
                    'relevance': evaluation.relevance,
                    'memorability': evaluation.memorability,
                    'brandability': evaluation.brandability,
                    'technical_quality': evaluation.technical_quality,
                    'creativity': evaluation.creativity,
                    'commercial_viability': evaluation.commercial_viability,
                    'overall_score': evaluation.overall_score,
                    'reasoning': evaluation.reasoning,
                    'improvement_suggestions': evaluation.improvement_suggestions,
                    'confidence': evaluation.confidence
                }
                
                results.append(case_with_eval)
                
                self.logger.info(f"Evaluated {i+1}/{total}: {generated} -> {evaluation.overall_score:.1f}/10")
                
                # Add delay to prevent overwhelming the model
                if i < total - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating case {i+1}: {e}")
                # Add case with empty evaluation
                case_with_eval = case.copy()
                case_with_eval['llm_evaluation'] = self._create_fallback_result(
                    case.get('business', ''), 
                    case.get('generated', '')
                ).__dict__
                results.append(case_with_eval)
        
        self.logger.info(f"Batch evaluation completed. {len(results)} results generated.")
        return results
    
    def save_evaluation_results(self, results: List[Dict], output_path: str):
        """Save evaluation results to JSON file"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test the LLM Judge
    judge = FreeLLMJudge(model_name="microsoft/DialoGPT-medium")
    
    # Test single evaluation
    test_business = "an AI assistant for coding"
    test_domain = "codecompanion.ai"
    
    print(f"Testing evaluation:")
    print(f"Business: {test_business}")
    print(f"Domain: {test_domain}")
    
    result = judge.evaluate_domain(test_business, test_domain)
    
    print(f"\nEvaluation Results:")
    print(f"Overall Score: {result.overall_score:.1f}/10")
    print(f"Relevance: {result.relevance}/10")
    print(f"Memorability: {result.memorability}/10")
    print(f"Reasoning: {result.reasoning}")
    print(f"Confidence: {result.confidence:.2f}")