# LLM-as-a-Judge Evaluation System

## Overview

This directory contains a comprehensive LLM-as-a-Judge evaluation system for domain name generation models. The system provides automated, multi-dimensional evaluation of generated domain names using free, open-source language models.

## 🎯 Evaluation Methodology

### Core Evaluation Metrics

Our LLM Judge evaluates domain names across **6 key dimensions** (1-10 scale):

1. **Relevance** - How well the domain relates to the business description
2. **Memorability** - Ease of remembering and pronouncing the domain
3. **Brandability** - Potential as a strong brand name
4. **Technical Quality** - Proper format, length, and extension appropriateness
5. **Creativity** - Uniqueness and innovation in the domain suggestion
6. **Commercial Viability** - Real-world business applicability

### Overall Scoring

- **Overall Score**: Weighted average of all 6 metrics
- **Confidence Score**: System confidence in the evaluation (0.0-1.0)
- **Reasoning**: Natural language explanation for the scores
- **Improvement Suggestions**: Actionable feedback for domain enhancement

## 📁 File Structure

```
src/evaluation/
├── llm_judge.py                    # Core LLM Judge implementation
├── model_comparison.py             # Statistical comparison framework
├── run_llm_judge_evaluation.py     # Command-line interface
├── baseline_llm_judge_colab.ipynb  # Interactive baseline evaluation
├── v1_1_llm_judge_colab.ipynb      # Interactive v1.1 evaluation
└── README.md                       # This documentation
```

## 🚀 Quick Start

### Option 1: Command Line Interface

```bash
# Evaluate baseline model
python src/evaluation/run_llm_judge_evaluation.py \
    --baseline-results evaluation_results/baseline_model/baseline_evaluation_results.csv \
    --output-dir evaluation_results/llm_judge

# Compare baseline vs improved model
python src/evaluation/run_llm_judge_evaluation.py \
    --baseline-results evaluation_results/baseline_model/baseline_evaluation_results.csv \
    --improved-results evaluation_results/v1_1_model/v1_1_evaluation_results.csv \
    --output-dir evaluation_results/llm_judge
```

### Option 2: Google Colab Notebooks

1. **For Baseline Model**: Open `baseline_llm_judge_colab.ipynb` in Google Colab
2. **For v1.1 Model**: Open `v1_1_llm_judge_colab.ipynb` in Google Colab
3. Upload your CSV files and run all cells

### Option 3: Python API

```python
from evaluation.llm_judge import FreeLLMJudge
from evaluation.model_comparison import ModelComparisonFramework

# Initialize judge
judge = FreeLLMJudge(model_name="microsoft/DialoGPT-medium")

# Single evaluation
result = judge.evaluate_domain("a coffee shop", "brewcafe.com")
print(f"Overall Score: {result.overall_score:.1f}/10")

# Batch evaluation
enhanced_results = judge.batch_evaluate(test_cases)

# Model comparison
comparison = ModelComparisonFramework()
comparison.add_model_results("baseline", baseline_results)
comparison.add_model_results("v1.1", improved_results)
```

## 🤖 Supported Models

### Free Open-Source Models (No API Costs)

| Model | Size | Speed | Quality | Recommended Use |
|-------|------|-------|---------|-----------------|
| `microsoft/DialoGPT-medium` | ~345MB | Fast | Good | Testing & development |
| `microsoft/DialoGPT-large` | ~774MB | Medium | Better | Production evaluation |
| `HuggingFaceH4/zephyr-7b-beta` | ~4GB | Slower | Excellent | Final evaluation |
| `mistralai/Mistral-7B-Instruct-v0.1` | ~4GB | Slower | Excellent | Research use |

### Model Selection Guide

- **Start with DialoGPT-medium** for quick testing
- **Use DialoGPT-large** for production evaluation
- **Use Zephyr/Mistral** for high-quality final evaluation (requires more GPU memory)

## 📊 Input/Output Format

### Input: CSV File Format

Your input CSV must contain these columns:
```csv
business,expected,generated,is_valid,similarity
"a coffee shop","brewcafe.com","coffeeshop.com",true,0.654
"a yoga studio","zenflow.com","yoga-place.com",false,0.234
```

### Output: Enhanced CSV Format

The system outputs enhanced CSV files with additional LLM judge columns:
```csv
business,expected,generated,is_valid,similarity,llm_relevance,llm_memorability,llm_brandability,llm_technical_quality,llm_creativity,llm_commercial_viability,llm_overall_score,llm_confidence,llm_reasoning
```

## 🔧 Configuration Options

### Command Line Arguments

```bash
--baseline-results     # Path to baseline evaluation CSV
--improved-results     # Path to improved model CSV (optional)
--output-dir          # Output directory for results
--judge-model         # LLM model to use for evaluation
--skip-judge          # Skip evaluation, use existing results
```

### Model Configuration

```python
# Initialize with custom settings
judge = FreeLLMJudge(
    model_name="microsoft/DialoGPT-large",
    device="cuda",  # "cpu", "cuda", or "auto"
    cache_dir="./models/cache"
)
```

## 📈 Evaluation Metrics Explained

### 1. Relevance (1-10)
- **High (8-10)**: Domain clearly relates to business (e.g., "brewcafe.com" for coffee shop)
- **Medium (5-7)**: Some connection but not obvious
- **Low (1-4)**: No clear relationship to business

### 2. Memorability (1-10)
- **High (8-10)**: Short, pronounceable, easy to remember
- **Medium (5-7)**: Reasonable length and pronunciation
- **Low (1-4)**: Long, complex, hard to remember

### 3. Brandability (1-10)
- **High (8-10)**: Strong brand potential, professional sound
- **Medium (5-7)**: Decent brand potential with some limitations
- **Low (1-4)**: Poor brand potential, unprofessional

### 4. Technical Quality (1-10)
- **High (8-10)**: Proper format, appropriate extension, good length
- **Medium (5-7)**: Minor technical issues
- **Low (1-4)**: Major format problems, inappropriate extension

### 5. Creativity (1-10)
- **High (8-10)**: Unique, innovative, stands out
- **Medium (5-7)**: Some creative elements
- **Low (1-4)**: Generic, common, unoriginal

### 6. Commercial Viability (1-10)
- **High (8-10)**: Strong business potential, market-ready
- **Medium (5-7)**: Decent commercial prospects
- **Low (1-4)**: Poor business viability

## 🔍 Understanding Confidence Scores

The confidence score indicates how reliable the LLM evaluation is:

- **0.8-1.0**: High confidence - clear JSON response, detailed reasoning
- **0.5-0.7**: Medium confidence - some parsing issues or brief reasoning
- **0.1-0.4**: Low confidence - fallback evaluation used, minimal reasoning

## 📊 Interpreting Results

### Score Ranges

- **8-10**: Excellent domain names ready for production use
- **6-8**: Good domains with minor improvement opportunities
- **4-6**: Average domains requiring significant enhancement
- **1-4**: Poor domains needing complete redesign

### Statistical Significance

When comparing models, look for:
- **p-value < 0.05**: Statistically significant improvement
- **Effect size > 0.5**: Practically meaningful difference
- **Confidence intervals**: Range of likely true performance

## 🛠 Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Try smaller model (DialoGPT-medium)
   - Clear cache directory

2. **GPU Memory Error**
   - Use CPU mode: `device="cpu"`
   - Try smaller model
   - Reduce batch size

3. **Import Errors**
   - Ensure you're in project root directory
   - Check Python path includes `src/`
   - Install required dependencies

4. **Evaluation Failures**
   - Check input CSV format
   - Verify column names match expected format
   - Use fallback evaluation if needed

### Performance Optimization

- **Use GPU** when available for faster evaluation
- **Batch processing** for multiple domains
- **Model caching** to avoid repeated downloads
- **Progress tracking** for long evaluations

## 📋 Dependencies

```bash
# Core requirements
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0

# Analysis and visualization
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0

# Export functionality
openpyxl>=3.1.0
```

## 🔄 Workflow Integration

### Step 1: Model Training
Train your domain generation model (baseline, v1.1, etc.)

### Step 2: Basic Evaluation
Run validity and similarity evaluation on test set

### Step 3: LLM Judge Evaluation
Enhance results with comprehensive LLM-based scoring

### Step 4: Model Comparison
Compare different model versions statistically

### Step 5: Report Generation
Generate visualizations and detailed comparison reports

## 📚 References

- **DialoGPT**: Microsoft's conversational AI model
- **Transformers**: Hugging Face transformer library
- **Statistical Testing**: Scipy t-tests for significance
- **Evaluation Methodology**: Multi-dimensional domain assessment

## 🤝 Contributing

When adding new evaluation metrics:
1. Update `JudgeResult` dataclass in `llm_judge.py`
2. Modify evaluation template prompt
3. Add corresponding parsing logic
4. Update visualization in `model_comparison.py`

## 📄 License

This evaluation system is designed for research and educational purposes. Model usage follows respective model licenses from Hugging Face.

---

**Last Updated**: August 2025 
**Version**: 1.0  
**Maintainer**: Wissal BENDIDI
