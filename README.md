# Domain Name Generator LLM

This project builds and iteratively improves an open-source language model 
to generate domain name suggestions for businesses.

**Features:**
- Synthetic dataset creation
- Fine-tuning open-source LLM
- Automated evaluation using GPT-4 as a judge
- Edge case discovery & analysis
- Safety guardrails to block inappropriate content
- (Optional) API endpoint for deployment

## Setup Instructions

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
