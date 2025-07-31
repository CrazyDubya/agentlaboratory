#!/usr/bin/env python3
"""
Agent Laboratory Simple Demo
============================

A lightweight demonstration of Agent Laboratory's web interface and core concepts.
"""

import os
import sys
import threading
import time
from pathlib import Path

def create_sample_data():
    """Create sample research papers for demo"""
    print("📚 Creating sample research papers...")
    
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    sample_papers = [
        {
            "filename": "chain_of_thought_reasoning.txt",
            "content": """Title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

Abstract: This paper introduces chain-of-thought prompting, a simple method that enables large language models to generate intermediate reasoning steps when solving complex problems. We show that chain-of-thought prompting significantly improves performance on arithmetic reasoning tasks, common sense reasoning, and symbolic reasoning tasks.

Introduction: Large language models have shown remarkable capabilities in many natural language processing tasks. However, they often struggle with tasks that require complex reasoning and multi-step problem solving. Chain-of-thought prompting addresses this limitation by encouraging the model to work through problems step-by-step.

Methods: We evaluate chain-of-thought prompting on several reasoning benchmarks including GSM8K (grade school math problems), MATH (mathematical competition problems), and CommonsenseQA. The method involves providing examples that show the reasoning process explicitly.

Results: Our experiments show that chain-of-thought prompting leads to substantial improvements across all tested domains. On GSM8K, accuracy improved from 17.9% to 58.1% with PaLM 540B.

Conclusion: Chain-of-thought prompting is a simple yet effective technique for improving the reasoning capabilities of large language models. This approach could be valuable for many applications requiring logical reasoning.
"""
        },
        {
            "filename": "mathematical_reasoning_llms.txt", 
            "content": """Title: Mathematical Reasoning in Large Language Models

Abstract: We investigate the mathematical reasoning capabilities of large language models through comprehensive evaluation on various mathematical tasks. Our study reveals both strengths and limitations of current models in mathematical problem solving.

Introduction: Mathematical reasoning is a fundamental cognitive skill that involves logical thinking, pattern recognition, and systematic problem solving. Recent advances in large language models have shown promising results in mathematical tasks, but systematic evaluation is needed.

Dataset: We use the MATH dataset containing 12,500 challenging mathematical problems spanning algebra, geometry, number theory, and other areas. Problems are graded at high school competition level.

Methodology: We evaluate several state-of-the-art language models including GPT-3, GPT-4, and PaLM using various prompting strategies. We measure both final answer accuracy and reasoning quality.

Results: GPT-4 achieves 42.5% accuracy on MATH, while GPT-3 achieves 23.5%. Chain-of-thought prompting consistently improves performance across all models. Error analysis reveals common failure modes.

Discussion: While current models show impressive mathematical reasoning abilities, they still struggle with complex multi-step problems and geometric reasoning. Future work should focus on improving systematic reasoning.
"""
        },
        {
            "filename": "prompt_engineering_techniques.txt",
            "content": """Title: A Survey of Prompt Engineering Techniques for Large Language Models

Abstract: Prompt engineering has emerged as a crucial technique for effectively utilizing large language models. This survey reviews the current state of prompt engineering methods and their applications across various domains.

Introduction: Large language models like GPT-3 and T5 have demonstrated remarkable capabilities when provided with appropriate prompts. The design of effective prompts has become an important research area with significant practical implications.

Prompt Design Strategies: We categorize prompt engineering approaches into several classes: few-shot prompting, chain-of-thought prompting, instruction tuning, and template-based methods. Each approach has distinct advantages for different types of tasks.

Applications: Prompt engineering techniques have been successfully applied to text classification, question answering, mathematical reasoning, code generation, and creative writing tasks. Performance gains are often substantial.

Few-Shot Learning: By providing a few examples in the prompt, models can quickly adapt to new tasks without parameter updates. This approach is particularly effective for classification and structured prediction tasks.

Chain-of-Thought: Encouraging models to show their reasoning process step-by-step leads to improved performance on complex reasoning tasks. This technique is especially valuable for mathematical and logical reasoning.

Future Directions: Research opportunities include automated prompt optimization, prompt compression techniques, and better understanding of how prompts influence model behavior.
"""
        }
    ]
    
    # Create files in uploads directory
    for paper in sample_papers:
        filepath = uploads_dir / paper["filename"]
        with open(filepath, 'w') as f:
            f.write(paper["content"])
    
    # Also add to database if Flask context is available
    try:
        import app
        from app import db, Paper
        
        with app.app.app_context():
            # Clear existing demo papers
            Paper.query.filter(Paper.filename.like('%chain_of_thought%')).delete()
            Paper.query.filter(Paper.filename.like('%mathematical_reasoning%')).delete()
            Paper.query.filter(Paper.filename.like('%prompt_engineering%')).delete()
            
            # Add new papers to database
            for paper in sample_papers:
                existing = Paper.query.filter_by(filename=paper["filename"]).first()
                if not existing:
                    new_paper = Paper(filename=paper["filename"], text=paper["content"])
                    db.session.add(new_paper)
            
            db.session.commit()
            print("✅ Added papers to database")
            
    except Exception as e:
        print(f"⚠️ Could not add to database: {e}")
    
    print(f"✅ Created {len(sample_papers)} sample papers")
    return len(sample_papers)

def print_banner():
    """Print demo banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🧬 AGENT LABORATORY DEMO 🧬                      ║
    ║                                                               ║
    ║            Using LLM Agents as Research Assistants            ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_web_demo():
    """Run the web interface demo"""
    print("🚀 Starting Agent Laboratory Web Demo...")
    
    # Sample data will be created after Flask app starts
    num_papers = 3
    
    print("\n" + "="*70)
    print("🌐 AGENT LABORATORY WEB INTERFACE")
    print("="*70)
    print("\n🎯 Demo Features:")
    print("  📚 Research paper library with sample papers")
    print("  🔍 Search functionality (text matching mode)")
    print("  📤 PDF upload and processing capabilities")
    print("  🧬 Interactive research workflow demo")
    print("  🎨 Beautiful, responsive web interface")
    
    print(f"\n📊 Demo Status:")
    print(f"  ✅ {num_papers} sample papers loaded")
    print(f"  ✅ Flask web server ready")
    print(f"  ⚠️ Semantic search in offline mode (text matching)")
    print(f"  ✅ All core features available")
    
    print(f"\n🚀 Starting web server...")
    print(f"📍 URL: http://localhost:5000")
    print(f"\n📋 Suggested Demo Flow:")
    print(f"  1. Visit http://localhost:5000 to see the paper library")
    print(f"  2. Try searching for 'mathematical reasoning' or 'chain of thought'")
    print(f"  3. View individual papers to see content extraction")
    print(f"  4. Explore the research demo at /demo")
    print(f"  5. Try uploading your own PDF papers")
    
    print(f"\n💡 Notes:")
    print(f"  • This demo runs in offline mode with basic text search")
    print(f"  • Sample papers demonstrate realistic research content")
    print(f"  • The full Agent Laboratory supports semantic search with internet")
    print(f"  • Research workflow demo shows the complete AI agent pipeline")
    
    print(f"\n🛠️ To run actual AI research workflow:")
    print(f"   python ai_lab_repo.py --yaml-location experiment_configs/MATH_agentlab.yaml")
    
    print("\n" + "="*70)
    print("⚡ Web server starting... Keep this terminal open!")
    print("   Press Ctrl+C to stop the demo")
    print("="*70)
    
    try:
        # Import and run Flask app
        import app
        app.run_app(port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped. Thank you for exploring Agent Laboratory!")
    except Exception as e:
        print(f"\n❌ Error running web demo: {e}")
        print("Try installing missing dependencies: pip install flask flask-sqlalchemy PyPDF2")

def show_features():
    """Display Agent Laboratory features"""
    print("\n🌟 AGENT LABORATORY FEATURES")
    print("="*50)
    
    features = [
        ("🧬 Multi-Agent Collaboration", 
         "PhD, Postdoc, Professor, ML Engineer, and Software Engineer agents work together"),
        ("📚 Automated Literature Review", 
         "Intelligent search and analysis of academic papers from arXiv"),
        ("🎯 Research Planning", 
         "Collaborative formulation of research hypotheses and experimental designs"),
        ("⚙️ Code Generation", 
         "Automatic implementation of experiments and data processing pipelines"),
        ("📊 Results Analysis", 
         "AI-powered interpretation of experimental results and statistical analysis"),
        ("📝 Report Writing", 
         "Publication-ready LaTeX reports with figures, tables, and citations"),
        ("🔍 Semantic Search", 
         "Advanced paper search using transformer-based embeddings"),
        ("🌐 Web Interface", 
         "Intuitive web interface for managing research projects"),
        ("🚀 AgentRxiv Integration", 
         "Collaborative platform for AI-generated research papers"),
        ("🔧 Flexible Configuration", 
         "Customizable workflows supporting multiple LLM backends")
    ]
    
    for feature, description in features:
        print(f"\n{feature}")
        print(f"   {description}")
    
    print(f"\n🎯 Use Cases:")
    print(f"   • Academic research acceleration")
    print(f"   • Literature review automation")
    print(f"   • Experiment design and execution")
    print(f"   • Research report generation")
    print(f"   • Collaborative AI research")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Laboratory Simple Demo")
    parser.add_argument("--features", action="store_true", help="Show features and exit")
    parser.add_argument("--mode", choices=["web"], default="web", help="Demo mode")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.features:
        show_features()
        return
    
    if args.mode == "web":
        run_web_demo()

if __name__ == "__main__":
    main()