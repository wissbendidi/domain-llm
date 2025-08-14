# src/evaluation/run_llm_judge_evaluation.py

"""
Integration script to run LLM-as-a-Judge evaluation on your existing model results
This script loads your baseline results and applies LLM judge evaluation
"""

import json
import pandas as pd
from pathlib import Path
import sys
import argparse
from typing import List, Dict
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from evaluation.llm_judge import FreeLLMJudge
    from evaluation.model_comparison import ModelComparisonFramework
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    print("Expected structure: src/evaluation/llm_judge.py")
    sys.exit(1)

def load_baseline_results(results_path: str) -> List[Dict]:
    """Load your existing baseline evaluation results"""
    
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    if results_path.suffix == '.csv':
        # Load CSV results (like your baseline_evaluation_results.csv)
        df = pd.read_csv(results_path)
        
        # Convert to list of dictionaries
        results = []
        for _, row in df.iterrows():
            results.append({
                'business': row['business'],
                'expected': row['expected'],
                'generated': row['generated'],
                'is_valid': row['is_valid'],
                'similarity': row['similarity']
            })
        return results
        
    elif results_path.suffix == '.json':
        # Load JSON results
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {results_path.suffix}")

def enhance_results_with_llm_judge(results: List[Dict], 
                                  model_name: str = "microsoft/DialoGPT-medium") -> List[Dict]:
    """Add LLM-as-a-Judge evaluation to existing results"""
    
    print(f"\n🤖 Initializing LLM Judge with model: {model_name}")
    print("📥 This will download the model if not already cached...")
    
    # Initialize LLM Judge
    judge = FreeLLMJudge(model_name=model_name)
    
    print(f"\n📊 Starting LLM Judge evaluation on {len(results)} cases...")
    
    def progress_callback(current, total):
        percent = (current / total) * 100
        print(f"Progress: {current}/{total} ({percent:.1f}%)")
    
    # Run batch evaluation
    enhanced_results = judge.batch_evaluate(
        results, 
        delay=0.3,  # Small delay to prevent overwhelming the model
        progress_callback=progress_callback
    )
    
    print("✅ LLM Judge evaluation completed!")
    return enhanced_results

def save_enhanced_results(results: List[Dict], 
                         output_dir: str, 
                         model_version: str):
    """Save the enhanced results with LLM judge scores"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_dir / f"{model_version}_with_llm_judge.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save as CSV for easy viewing
    csv_path = output_dir / f"{model_version}_with_llm_judge.csv"
    
    # Flatten the data for CSV
    flattened_data = []
    for result in results:
        flat_result = {
            'business': result['business'],
            'expected': result['expected'],
            'generated': result['generated'],
            'is_valid': result['is_valid'],
            'similarity': result['similarity']
        }
        
        # Add LLM judge scores if present
        if 'llm_evaluation' in result:
            llm_eval = result['llm_evaluation']
            flat_result.update({
                'llm_relevance': llm_eval['relevance'],
                'llm_memorability': llm_eval['memorability'],
                'llm_brandability': llm_eval['brandability'],
                'llm_technical_quality': llm_eval['technical_quality'],
                'llm_creativity': llm_eval['creativity'],
                'llm_commercial_viability': llm_eval['commercial_viability'],
                'llm_overall_score': llm_eval['overall_score'],
                'llm_confidence': llm_eval['confidence'],
                'llm_reasoning': llm_eval['reasoning'][:100] + "..." if len(llm_eval['reasoning']) > 100 else llm_eval['reasoning']
            })
        
        flattened_data.append(flat_result)
    
    df = pd.DataFrame(flattened_data)
    df.to_csv(csv_path, index=False)
    
    print(f"📁 Enhanced results saved:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    
    return json_path, csv_path

def generate_comparison_report(baseline_results: List[Dict], 
                              improved_results: List[Dict] = None,
                              output_dir: str = "evaluation_results"):
    """Generate a comparison report using the ModelComparisonFramework"""
    
    print("\n📊 Generating comparison report...")
    
    try:
        # Initialize comparison framework
        comparison = ModelComparisonFramework(output_dir=output_dir)
        
        # Add baseline results
        comparison.add_model_results(
            "baseline", 
            baseline_results, 
            "TinyLlama baseline model with basic configuration"
        )
        
        # Add improved results if available
        if improved_results:
            comparison.add_model_results(
                "v1.1", 
                improved_results, 
                "Enhanced model with improved parameters"
            )
        
        # Generate report
        if len(comparison.model_results) >= 2:
            report = comparison.generate_detailed_report("model_comparison_report.md")
            print("✅ Comparison report generated!")
            
            # Create visualization dashboard
            dashboard_path = Path(output_dir) / "performance_dashboard.png"
            comparison.create_visualization_dashboard(str(dashboard_path))
            
            # Export results
            comparison.export_results("excel")
            
        else:
            print("ℹ️  Only one model available - skipping comparison")
        
        return comparison
    
    except ImportError:
        print("⚠️  ModelComparisonFramework not available - skipping comparison")
        return None

def print_summary_statistics(results: List[Dict], model_name: str):
    """Print summary statistics for the enhanced results"""
    
    print(f"\n📈 {model_name.upper()} MODEL - LLM JUDGE RESULTS SUMMARY")
    print("=" * 60)
    
    # Basic statistics
    total_cases = len(results)
    valid_domains = sum(1 for r in results if r.get('is_valid', False))
    validity_rate = valid_domains / total_cases if total_cases > 0 else 0
    
    similarity_scores = [r.get('similarity', 0) for r in results]
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    print(f"📊 Total test cases: {total_cases}")
    print(f"✅ Valid domains: {valid_domains} ({validity_rate:.1%})")
    print(f"🎯 Average similarity: {avg_similarity:.3f}")
    
    # LLM Judge statistics
    llm_results = [r['llm_evaluation'] for r in results if 'llm_evaluation' in r]
    
    if llm_results:
        print(f"\n🤖 LLM JUDGE EVALUATION ({len(llm_results)} cases):")
        print("-" * 40)
        
        metrics = ['relevance', 'memorability', 'brandability', 
                  'technical_quality', 'creativity', 'commercial_viability', 'overall_score']
        
        for metric in metrics:
            scores = [lr[metric] for lr in llm_results]
            avg_score = sum(scores) / len(scores)
            print(f"📊 {metric.replace('_', ' ').title():<20}: {avg_score:.1f}/10")
        
        # Confidence
        confidence_scores = [lr['confidence'] for lr in llm_results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"🎯 Average Confidence{'':<7}: {avg_confidence:.2f}")
        
        # Score distribution
        overall_scores = [lr['overall_score'] for lr in llm_results]
        excellent = sum(1 for s in overall_scores if s >= 8)
        good = sum(1 for s in overall_scores if 6 <= s < 8)
        poor = sum(1 for s in overall_scores if s < 6)
        
        print(f"\n📈 SCORE DISTRIBUTION:")
        print(f"   🌟 Excellent (8-10): {excellent} ({excellent/len(overall_scores):.1%})")
        print(f"   👍 Good (6-8):       {good} ({good/len(overall_scores):.1%})")
        print(f"   👎 Poor (<6):        {poor} ({poor/len(overall_scores):.1%})")
        
        # Best and worst examples
        sorted_results = sorted(results, key=lambda x: x.get('llm_evaluation', {}).get('overall_score', 0), reverse=True)
        
        print(f"\n🏆 TOP 3 DOMAINS:")
        for i, result in enumerate(sorted_results[:3]):
            if 'llm_evaluation' in result:
                llm_eval = result['llm_evaluation']
                print(f"   {i+1}. {result['generated']} -> {llm_eval['overall_score']:.1f}/10")
                print(f"      Business: {result['business']}")
        
        print(f"\n👎 BOTTOM 3 DOMAINS:")
        for i, result in enumerate(sorted_results[-3:]):
            if 'llm_evaluation' in result:
                llm_eval = result['llm_evaluation']
                print(f"   {i+1}. {result['generated']} -> {llm_eval['overall_score']:.1f}/10")
                print(f"      Business: {result['business']}")

def main():
    """Main function to run LLM Judge evaluation"""
    
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge evaluation on model results")
    parser.add_argument("--baseline-results", 
                       default="evaluation_results/baseline_model/baseline_evaluation_results.csv",
                       help="Path to baseline evaluation results")
    parser.add_argument("--improved-results", 
                       default=None,
                       help="Path to improved model results (optional)")
    parser.add_argument("--output-dir", 
                       default="evaluation_results/llm_judge",
                       help="Output directory for enhanced results")
    parser.add_argument("--judge-model", 
                       default="microsoft/DialoGPT-medium",
                       choices=["microsoft/DialoGPT-medium", 
                               "microsoft/DialoGPT-large",
                               "HuggingFaceH4/zephyr-7b-beta",
                               "mistralai/Mistral-7B-Instruct-v0.1"],
                       help="LLM model to use for judging")
    parser.add_argument("--skip-judge", 
                       action="store_true",
                       help="Skip LLM judge evaluation (use existing results)")
    
    args = parser.parse_args()
    
    print("🚀 LLM-as-a-Judge Evaluation Pipeline")
    print("=" * 50)
    
    try:
        # Load baseline results
        print(f"📂 Loading baseline results from: {args.baseline_results}")
        baseline_results = load_baseline_results(args.baseline_results)
        print(f"✅ Loaded {len(baseline_results)} baseline results")
        
        # Enhance with LLM judge if not skipping
        if not args.skip_judge:
            baseline_results = enhance_results_with_llm_judge(baseline_results, args.judge_model)
            
            # Save enhanced baseline results
            save_enhanced_results(baseline_results, args.output_dir, "baseline")
        
        # Print baseline summary
        print_summary_statistics(baseline_results, "baseline")
        
        # Load improved results if available
        improved_results = None
        if args.improved_results:
            print(f"\n📂 Loading improved model results from: {args.improved_results}")
            improved_results = load_baseline_results(args.improved_results)
            print(f"✅ Loaded {len(improved_results)} improved results")
            
            if not args.skip_judge:
                improved_results = enhance_results_with_llm_judge(improved_results, args.judge_model)
                save_enhanced_results(improved_results, args.output_dir, "improved")
            
            print_summary_statistics(improved_results, "improved")
        
        # Generate comparison report
        comparison_framework = generate_comparison_report(
            baseline_results, 
            improved_results, 
            args.output_dir
        )
        
        print(f"\n🎉 LLM Judge evaluation completed!")
        print(f"📁 Results saved in: {args.output_dir}")
        
        if improved_results and comparison_framework:
            print(f"\n🏆 RECOMMENDATION:")
            rankings = comparison_framework.compare_models()['overall_ranking']
            best_model = rankings[0]
            print(f"   Best performing model: {best_model['model']} (Score: {best_model['composite_score']:.1f}/10)")
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Make sure the file path is correct and the file exists")
        return 1
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())