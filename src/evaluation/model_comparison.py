# src/evaluation/model_comparison.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path
from scipy import stats
import logging

class ModelComparisonFramework:
    """
    Framework for comparing different model versions using LLM-as-a-Judge evaluation
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store model results
        self.model_results = {}
        
    def add_model_results(self, 
                         model_name: str, 
                         results: List[Dict], 
                         model_description: str = ""):
        """
        Add evaluation results for a model version
        
        Args:
            model_name: Name/version of the model (e.g., "baseline", "v1.1")
            results: List of evaluation results with LLM judge scores
            model_description: Optional description of the model
        """
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(results)
        
        self.model_results[model_name] = {
            'description': model_description,
            'results': results,
            'summary': summary_stats,
            'total_cases': len(results)
        }
        
        self.logger.info(f"Added results for {model_name}: {len(results)} cases")
    
    def _calculate_summary_stats(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics from results"""
        
        # Basic metrics
        total_cases = len(results)
        valid_domains = sum(1 for r in results if r.get('is_valid', False))
        validity_rate = valid_domains / total_cases if total_cases > 0 else 0
        
        # Similarity scores
        similarity_scores = [r.get('similarity', 0) for r in results]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        std_similarity = np.std(similarity_scores) if similarity_scores else 0
        
        # LLM Judge scores
        llm_metrics = {}
        if any('llm_evaluation' in r for r in results):
            judge_results = [r['llm_evaluation'] for r in results if 'llm_evaluation' in r]
            
            metrics = ['relevance', 'memorability', 'brandability', 
                      'technical_quality', 'creativity', 'commercial_viability', 'overall_score']
            
            for metric in metrics:
                scores = [jr.get(metric, 0) for jr in judge_results]
                if scores:
                    llm_metrics[f'avg_{metric}'] = np.mean(scores)
                    llm_metrics[f'std_{metric}'] = np.std(scores)
                    llm_metrics[f'min_{metric}'] = np.min(scores)
                    llm_metrics[f'max_{metric}'] = np.max(scores)
        
        # Confidence scores
        confidence_scores = []
        if any('llm_evaluation' in r for r in results):
            confidence_scores = [r['llm_evaluation'].get('confidence', 0) 
                               for r in results if 'llm_evaluation' in r]
        
        return {
            'total_cases': total_cases,
            'validity_rate': validity_rate,
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            **llm_metrics
        }
    
    def compare_models(self) -> Dict:
        """
        Perform comprehensive comparison between all loaded models
        
        Returns:
            Dictionary with comparison results
        """
        
        if len(self.model_results) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        model_names = list(self.model_results.keys())
        comparisons = {}
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                comparisons[comparison_key] = self._compare_two_models(model1, model2)
        
        # Overall ranking
        overall_ranking = self._calculate_overall_ranking()
        
        return {
            'pairwise_comparisons': comparisons,
            'overall_ranking': overall_ranking,
            'summary_table': self._create_summary_table()
        }
    
    def _compare_two_models(self, model1: str, model2: str) -> Dict:
        """Compare two specific models"""
        
        data1 = self.model_results[model1]['summary']
        data2 = self.model_results[model2]['summary']
        
        # Statistical significance tests
        results1 = self.model_results[model1]['results']
        results2 = self.model_results[model2]['results']
        
        # Validity rate comparison
        valid1 = sum(1 for r in results1 if r.get('is_valid', False))
        valid2 = sum(1 for r in results2 if r.get('is_valid', False))
        
        # Similarity comparison
        sim1 = [r.get('similarity', 0) for r in results1]
        sim2 = [r.get('similarity', 0) for r in results2]
        
        # LLM Judge overall score comparison
        overall1 = [r['llm_evaluation']['overall_score'] 
                   for r in results1 if 'llm_evaluation' in r]
        overall2 = [r['llm_evaluation']['overall_score'] 
                   for r in results2 if 'llm_evaluation' in r]
        
        # Statistical tests
        similarity_test = self._perform_significance_test(sim1, sim2)
        overall_test = self._perform_significance_test(overall1, overall2) if overall1 and overall2 else None
        
        # Calculate improvements
        improvements = {}
        metrics_to_compare = [
            ('validity_rate', 'Validity Rate'),
            ('avg_similarity', 'Average Similarity'),
            ('avg_overall_score', 'LLM Judge Overall Score'),
            ('avg_relevance', 'Business Relevance'),
            ('avg_memorability', 'Memorability'),
            ('avg_creativity', 'Creativity')
        ]
        
        for metric, description in metrics_to_compare:
            val1 = data1.get(metric, 0)
            val2 = data2.get(metric, 0)
            
            if val1 > 0:  # Avoid division by zero
                improvement = ((val2 - val1) / val1) * 100
                improvements[metric] = {
                    'description': description,
                    'model1_value': val1,
                    'model2_value': val2,
                    'improvement_pct': improvement,
                    'better_model': model2 if val2 > val1 else model1
                }
        
        return {
            'improvements': improvements,
            'statistical_tests': {
                'similarity': similarity_test,
                'overall_score': overall_test
            },
            'winner': self._determine_winner(data1, data2),
            'summary': self._create_comparison_summary(model1, model2, improvements)
        }
    
    def _perform_significance_test(self, data1: List[float], data2: List[float]) -> Dict:
        """Perform statistical significance test"""
        
        try:
            if len(data1) > 1 and len(data2) > 1:
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                return {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'effect_size': (np.mean(data2) - np.mean(data1)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
                }
            else:
                return {'error': 'Insufficient data for statistical test'}
                
        except Exception as e:
            return {'error': f'Statistical test failed: {str(e)}'}
    
    def _determine_winner(self, data1: Dict, data2: Dict) -> str:
        """Determine overall winner between two models"""
        
        # Weighted scoring system
        weights = {
            'validity_rate': 0.3,
            'avg_similarity': 0.2,
            'avg_overall_score': 0.3,
            'avg_relevance': 0.1,
            'avg_memorability': 0.1
        }
        
        score1 = score2 = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            val1 = data1.get(metric, 0)
            val2 = data2.get(metric, 0)
            
            if val1 > 0 or val2 > 0:  # Only include if at least one model has data
                total_weight += weight
                if val1 > val2:
                    score1 += weight
                elif val2 > val1:
                    score2 += weight
        
        if total_weight == 0:
            return "Tie (no comparable data)"
        
        if score1 > score2:
            return "Model 1"
        elif score2 > score1:
            return "Model 2"
        else:
            return "Tie"
    
    def _calculate_overall_ranking(self) -> List[Dict]:
        """Calculate overall ranking of all models"""
        
        rankings = []
        
        for model_name, model_data in self.model_results.items():
            summary = model_data['summary']
            
            # Calculate composite score
            weights = {
                'validity_rate': 0.3,
                'avg_similarity': 0.2,
                'avg_overall_score': 0.3,
                'avg_relevance': 0.1,
                'avg_memorability': 0.1
            }
            
            composite_score = 0
            total_weight = 0
            
            for metric, weight in weights.items():
                value = summary.get(metric, 0)
                if metric == 'validity_rate' or metric.startswith('avg_similarity'):
                    # These are 0-1 scale, convert to 0-10
                    value *= 10
                
                if value > 0:
                    composite_score += value * weight
                    total_weight += weight
            
            if total_weight > 0:
                composite_score /= total_weight
            
            rankings.append({
                'model': model_name,
                'composite_score': composite_score,
                'validity_rate': summary.get('validity_rate', 0),
                'avg_similarity': summary.get('avg_similarity', 0),
                'avg_overall_score': summary.get('avg_overall_score', 0),
                'description': model_data['description']
            })
        
        # Sort by composite score
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add rank
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _create_summary_table(self) -> pd.DataFrame:
        """Create summary table of all models"""
        
        summary_data = []
        
        for model_name, model_data in self.model_results.items():
            summary = model_data['summary']
            
            summary_data.append({
                'Model': model_name,
                'Total Cases': summary.get('total_cases', 0),
                'Validity Rate': f"{summary.get('validity_rate', 0):.1%}",
                'Avg Similarity': f"{summary.get('avg_similarity', 0):.3f}",
                'LLM Overall Score': f"{summary.get('avg_overall_score', 0):.1f}/10",
                'Relevance': f"{summary.get('avg_relevance', 0):.1f}/10",
                'Memorability': f"{summary.get('avg_memorability', 0):.1f}/10",
                'Creativity': f"{summary.get('avg_creativity', 0):.1f}/10",
                'Confidence': f"{summary.get('avg_confidence', 0):.2f}",
                'Description': model_data['description'][:50] + "..." if len(model_data['description']) > 50 else model_data['description']
            })
        
        return pd.DataFrame(summary_data)
    
    def _create_comparison_summary(self, model1: str, model2: str, improvements: Dict) -> str:
        """Create human-readable comparison summary"""
        
        significant_improvements = []
        for metric, data in improvements.items():
            if abs(data['improvement_pct']) > 5:  # Only report >5% changes
                direction = "improved" if data['improvement_pct'] > 0 else "decreased"
                significant_improvements.append(
                    f"{data['description']}: {direction} by {abs(data['improvement_pct']):.1f}%"
                )
        
        if significant_improvements:
            return f"{model2} vs {model1}: " + "; ".join(significant_improvements)
        else:
            return f"{model2} vs {model1}: No significant improvements detected"
    
    def create_visualization_dashboard(self, save_path: str = None):
        """Create comprehensive visualization dashboard"""
        
        if len(self.model_results) < 2:
            print("Need at least 2 models for visualization")
            return
        
        # Setup the plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # Prepare data
        models_data = []
        for model_name, model_info in self.model_results.items():
            summary = model_info['summary'].copy()
            summary['model'] = model_name
            models_data.append(summary)
        
        df = pd.DataFrame(models_data)
        
        # 1. Validity Rate Comparison
        if 'validity_rate' in df.columns:
            bars1 = axes[0,0].bar(df['model'], df['validity_rate'] * 100)
            axes[0,0].set_title('Domain Validity Rate (%)')
            axes[0,0].set_ylabel('Validity Rate (%)')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Similarity Score Comparison  
        if 'avg_similarity' in df.columns:
            bars2 = axes[0,1].bar(df['model'], df['avg_similarity'])
            axes[0,1].set_title('Average Similarity Score')
            axes[0,1].set_ylabel('Similarity Score (0-1)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            for bar in bars2:
                height = bar.get_height()
                axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}', ha='center', va='bottom')
        
        # 3. LLM Judge Overall Score
        if 'avg_overall_score' in df.columns:
            bars3 = axes[0,2].bar(df['model'], df['avg_overall_score'])
            axes[0,2].set_title('LLM Judge Overall Score')
            axes[0,2].set_ylabel('Score (1-10)')
            axes[0,2].tick_params(axis='x', rotation=45)
            axes[0,2].set_ylim(0, 10)
            
            for bar in bars3:
                height = bar.get_height()
                axes[0,2].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1f}', ha='center', va='bottom')
        
        # 4. Radar Chart for Multiple Metrics
        self._create_radar_chart(axes[1,0], df)
        
        # 5. Score Distribution Comparison
        self._create_score_distribution(axes[1,1])
        
        # 6. Improvement Heatmap
        self._create_improvement_heatmap(axes[1,2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dashboard saved to {save_path}")
        
        plt.show()
    
    def _create_radar_chart(self, ax, df):
        """Create radar chart comparing multiple metrics"""
        
        try:
            from math import pi
            
            # Metrics for radar chart
            metrics = ['avg_relevance', 'avg_memorability', 'avg_brandability', 
                      'avg_creativity', 'avg_technical_quality', 'avg_commercial_viability']
            
            # Check if we have the required metrics
            available_metrics = [m for m in metrics if m in df.columns]
            
            if len(available_metrics) < 3:
                ax.text(0.5, 0.5, 'LLM Judge metrics\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('LLM Judge Metrics (Radar Chart)')
                return
            
            # Number of variables
            N = len(available_metrics)
            
            # Angle of each axis
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Colors for different models
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (_, row) in enumerate(df.iterrows()):
                if i >= len(colors):
                    break
                    
                # Values for this model
                values = [row.get(metric, 0) for metric in available_metrics]
                values += values[:1]  # Complete the circle
                
                # Plot
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=row['model'], color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('avg_', '').title() for m in available_metrics])
            ax.set_ylim(0, 10)
            ax.set_title('LLM Judge Metrics Comparison')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Radar chart error:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('LLM Judge Metrics (Error)')
    
    def _create_score_distribution(self, ax):
        """Create score distribution comparison"""
        
        try:
            for model_name, model_data in self.model_results.items():
                # Get overall scores from LLM judge
                overall_scores = []
                for result in model_data['results']:
                    if 'llm_evaluation' in result:
                        overall_scores.append(result['llm_evaluation']['overall_score'])
                
                if overall_scores:
                    ax.hist(overall_scores, bins=10, alpha=0.6, label=model_name, density=True)
            
            ax.set_title('LLM Judge Score Distribution')
            ax.set_xlabel('Overall Score (1-10)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Score distribution\nerror: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Score Distribution (Error)')
    
    def _create_improvement_heatmap(self, ax):
        """Create improvement heatmap between models"""
        
        try:
            if len(self.model_results) < 2:
                ax.text(0.5, 0.5, 'Need 2+ models\nfor comparison', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Create improvement matrix
            models = list(self.model_results.keys())
            metrics = ['validity_rate', 'avg_similarity', 'avg_overall_score']
            
            improvement_matrix = np.zeros((len(models), len(metrics)))
            
            for i, model in enumerate(models):
                summary = self.model_results[model]['summary']
                for j, metric in enumerate(metrics):
                    value = summary.get(metric, 0)
                    if metric == 'validity_rate':
                        value *= 100  # Convert to percentage
                    elif metric == 'avg_similarity':
                        value *= 10   # Scale to 0-10
                    improvement_matrix[i, j] = value
            
            # Create heatmap
            im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m.replace('avg_', '').replace('_', ' ').title() for m in metrics])
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models)
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(metrics)):
                    text = ax.text(j, i, f'{improvement_matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black")
            
            ax.set_title('Model Performance Heatmap')
            plt.colorbar(im, ax=ax)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Heatmap error:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Heatmap (Error)')
    
    def generate_detailed_report(self, output_file: str = None) -> str:
        """Generate a comprehensive comparison report"""
        
        if len(self.model_results) < 2:
            return "Need at least 2 models for comparison report"
        
        # Perform comparison
        comparison_results = self.compare_models()
        
        report = []
        report.append("# Model Comparison Report")
        report.append("=" * 50)
        report.append("")
        
        # Overall ranking
        report.append("## Overall Model Ranking")
        report.append("")
        rankings = comparison_results['overall_ranking']
        
        for rank_data in rankings:
            report.append(f"**{rank_data['rank']}. {rank_data['model']}** "
                         f"(Score: {rank_data['composite_score']:.1f}/10)")
            report.append(f"   - Validity Rate: {rank_data['validity_rate']:.1%}")
            report.append(f"   - Avg Similarity: {rank_data['avg_similarity']:.3f}")
            report.append(f"   - LLM Judge Score: {rank_data['avg_overall_score']:.1f}/10")
            report.append(f"   - Description: {rank_data['description']}")
            report.append("")
        
        # Pairwise comparisons
        report.append("## Detailed Pairwise Comparisons")
        report.append("")
        
        for comparison_name, comparison_data in comparison_results['pairwise_comparisons'].items():
            report.append(f"### {comparison_name}")
            report.append("")
            report.append(f"**Winner:** {comparison_data['winner']}")
            report.append("")
            report.append("**Key Improvements:**")
            
            for metric, data in comparison_data['improvements'].items():
                if abs(data['improvement_pct']) > 1:  # Only show meaningful changes
                    direction = "↗️" if data['improvement_pct'] > 0 else "↘️"
                    report.append(f"- {data['description']}: {direction} {data['improvement_pct']:.1f}%")
            
            report.append("")
            report.append(f"**Summary:** {comparison_data['summary']}")
            report.append("")
        
        # Statistical significance
        report.append("## Statistical Significance")
        report.append("")
        
        for comparison_name, comparison_data in comparison_results['pairwise_comparisons'].items():
            similarity_test = comparison_data['statistical_tests']['similarity']
            overall_test = comparison_data['statistical_tests']['overall_score']
            
            if similarity_test and 'p_value' in similarity_test:
                sig_marker = "✅" if similarity_test['significant'] else "❌"
                report.append(f"**{comparison_name} - Similarity:** {sig_marker} "
                             f"p-value = {similarity_test['p_value']:.4f}")
            
            if overall_test and 'p_value' in overall_test:
                sig_marker = "✅" if overall_test['significant'] else "❌"
                report.append(f"**{comparison_name} - LLM Judge:** {sig_marker} "
                             f"p-value = {overall_test['p_value']:.4f}")
        
        report.append("")
        
        # Summary table
        report.append("## Summary Table")
        report.append("")
        summary_df = comparison_results['summary_table']
        report.append(summary_df.to_string(index=False))
        
        # Join all parts
        full_report = "\n".join(report)
        
        # Save to file if requested
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_report)
            self.logger.info(f"Report saved to {output_path}")
        
        return full_report
    
    def export_results(self, format: str = "json"):
        """Export all results and comparisons"""
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            # Export to JSON
            export_data = {
                'timestamp': timestamp,
                'model_results': self.model_results,
                'comparison': self.compare_models() if len(self.model_results) >= 2 else None
            }
            
            output_path = self.output_dir / f"model_comparison_{timestamp}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
        elif format.lower() == "excel":
            # Export to Excel with multiple sheets
            output_path = self.output_dir / f"model_comparison_{timestamp}.xlsx"
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary table
                if len(self.model_results) >= 2:
                    summary_df = self.compare_models()['summary_table']
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual model details
                for model_name, model_data in self.model_results.items():
                    # Create DataFrame from results
                    results_df = pd.DataFrame(model_data['results'])
                    
                    # Flatten LLM evaluation if present
                    if 'llm_evaluation' in results_df.columns:
                        llm_df = pd.json_normalize(results_df['llm_evaluation'])
                        results_df = pd.concat([results_df.drop('llm_evaluation', axis=1), llm_df], axis=1)
                    
                    sheet_name = model_name[:31]  # Excel sheet name limit
                    results_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        self.logger.info(f"Results exported to {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    # Example of how to use the comparison framework
    comparison_framework = ModelComparisonFramework()
    
    # You would load your actual results here
    # comparison_framework.add_model_results("baseline", baseline_results, "Basic baseline model")
    # comparison_framework.add_model_results("v1.1", improved_results, "Enhanced model with better parameters")
    
    # Generate comparison
    # comparison_results = comparison_framework.compare_models()
    # print(comparison_framework.generate_detailed_report())
    
    print("Model comparison framework ready for use!")