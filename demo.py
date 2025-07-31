#!/usr/bin/env python3
"""
Agent Laboratory Demo Script
============================

This script provides a comprehensive demonstration of Agent Laboratory's capabilities,
showcasing the full research workflow from literature review to report generation.

Usage:
    python demo.py [--mode web|cli|full] [--topic "research topic"] [--quick]

Modes:
    web  - Launch web interface demo (default)
    cli  - Command-line interactive demo
    full - Complete research workflow (requires API keys)

Example:
    python demo.py --mode web
    python demo.py --mode cli --topic "novel prompting techniques for mathematical reasoning"
    python demo.py --mode full --topic "improving language model performance"
"""

import os
import sys
import argparse
import threading
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import app
    from common_imports import *
    from agents import *
    from ai_lab_repo import LaboratoryWorkflow
    print("✅ Agent Laboratory modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

class AgentLabDemo:
    def __init__(self):
        self.demo_data_dir = Path("demo_data")
        self.demo_data_dir.mkdir(exist_ok=True)
        self.sample_papers = [
            {
                "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                "content": "This paper introduces chain-of-thought prompting, a technique that enables large language models to perform complex reasoning tasks by generating intermediate reasoning steps..."
            },
            {
                "title": "Mathematical Reasoning via Self-Supervised Learning",
                "content": "We propose a novel approach for improving mathematical reasoning capabilities in language models through self-supervised learning techniques and structured data augmentation..."
            },
            {
                "title": "Prompt Engineering for Few-Shot Learning",
                "content": "This work explores systematic approaches to prompt engineering that significantly improve few-shot learning performance across various natural language understanding tasks..."
            }
        ]
        
    def print_banner(self):
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
        
    def create_sample_data(self):
        """Create sample research papers for demo"""
        print("\n📚 Creating sample research papers...")
        
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Create sample PDFs with dummy content for demo
        for i, paper in enumerate(self.sample_papers):
            filename = f"sample_paper_{i+1}.txt"
            filepath = uploads_dir / filename
            with open(filepath, 'w') as f:
                f.write(f"Title: {paper['title']}\n\n")
                f.write(paper['content'])
                f.write("\n\n[This is a demo file created for Agent Laboratory demonstration]")
        
        print(f"✅ Created {len(self.sample_papers)} sample papers in uploads/")
        
    def demo_web_interface(self):
        """Launch web interface demo"""
        print("\n🚀 Launching Agent Laboratory Web Interface...")
        print("\nFeatures demonstrated:")
        print("  📚 Research paper library management")
        print("  🔍 Semantic search capabilities")
        print("  📤 PDF upload and processing")
        print("  🧬 Interactive research demo")
        
        self.create_sample_data()
        
        def run_flask():
            try:
                app.run_app(port=5000)
            except Exception as e:
                print(f"Error running Flask app: {e}")
        
        # Start Flask in a separate thread
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        time.sleep(3)  # Give Flask time to start
        
        print("\n" + "="*60)
        print("🌐 Web Interface Running!")
        print("="*60)
        print(f"📍 URL: http://localhost:5000")
        print("\n📋 Demo Tour:")
        print("  1. Visit the main page to see the paper library")
        print("  2. Try uploading a PDF research paper")
        print("  3. Use the search function to find relevant papers")
        print("  4. Explore the interactive research demo")
        print("\n⚡ Demo Highlights:")
        print("  • Semantic search using sentence transformers")
        print("  • PDF text extraction and indexing")
        print("  • Beautiful, responsive web interface")
        print("  • Interactive research workflow showcase")
        print("\n💡 Tip: Keep this terminal open to see server logs")
        print("Press Ctrl+C to stop the demo server")
        print("="*60)
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Demo stopped. Thank you for exploring Agent Laboratory!")
            
    def demo_cli_interface(self, topic=None):
        """Command-line interactive demo"""
        print("\n💻 Agent Laboratory CLI Demo")
        print("="*50)
        
        if not topic:
            print("\n🎯 Research Topic Selection")
            print("Choose a research area to explore:")
            print("  1. Mathematical reasoning and prompting techniques")
            print("  2. Natural language processing optimization")
            print("  3. Machine learning model evaluation")
            print("  4. Custom topic")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            topics = {
                "1": "Novel prompting techniques for improving mathematical reasoning in large language models",
                "2": "Optimization strategies for natural language processing tasks using transformer architectures",
                "3": "Comprehensive evaluation methodologies for machine learning model performance",
                "4": None
            }
            
            topic = topics.get(choice)
            if topic is None and choice == "4":
                topic = input("Enter your custom research topic: ").strip()
            elif topic is None:
                topic = topics["1"]  # Default
        
        print(f"\n🔬 Selected Research Topic:")
        print(f"   {topic}")
        
        print("\n🤖 Agent Laboratory Workflow Simulation")
        print("="*50)
        
        # Simulate the research workflow
        phases = [
            ("📚 Literature Review", "Searching arXiv and academic databases for relevant papers..."),
            ("🎯 Plan Formulation", "PhD and Postdoc agents collaborating on research design..."),
            ("⚙️ Data Preparation", "ML Engineer preparing datasets and experimental setup..."),
            ("🔬 Running Experiments", "Executing experiments with parallelized processing..."),
            ("📊 Results Analysis", "Postdoc agent interpreting experimental results..."),
            ("📝 Report Writing", "Generating comprehensive research report with LaTeX..."),
            ("👥 Peer Review", "Multiple reviewer agents evaluating the research...")
        ]
        
        for phase_name, description in phases:
            print(f"\n{phase_name}")
            print(f"   {description}")
            
            # Simulate processing time
            for i in range(3):
                print("   " + "." * (i + 1), end="\r")
                time.sleep(0.5)
            print("   ✅ Complete")
        
        print("\n🎉 Research Workflow Complete!")
        print("="*50)
        print("\n📋 Demo Summary:")
        print("  • Demonstrated multi-agent collaboration")
        print("  • Showed complete research pipeline")
        print("  • Simulated autonomous research execution")
        print("  • Highlighted key Agent Laboratory features")
        
        print(f"\n💡 To run the actual research workflow:")
        print(f"   python ai_lab_repo.py --yaml-location experiment_configs/MATH_agentlab.yaml")
        
    def demo_full_workflow(self, topic, quick_mode=False):
        """Run actual Agent Laboratory workflow (requires API keys)"""
        print("\n🚀 Running Full Agent Laboratory Workflow")
        print("="*50)
        
        # Check for API keys
        api_key = os.getenv('OPENAI_API_KEY')
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        if not api_key and not deepseek_key:
            print("❌ No API keys found!")
            print("Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable")
            print("\nExample:")
            print("  export OPENAI_API_KEY='your-key-here'")
            print("  # or")
            print("  export DEEPSEEK_API_KEY='your-key-here'")
            return
            
        print(f"✅ API key detected")
        print(f"🎯 Research Topic: {topic}")
        
        # Configure workflow parameters
        notes = [
            {"phases": ["plan formulation"], "note": f"Focus on: {topic}"},
            {"phases": ["data preparation"], "note": "Use efficient data processing techniques"},
            {"phases": ["running experiments"], "note": "Ensure reproducible results"},
        ]
        
        human_in_loop = {
            "literature review": False,
            "plan formulation": False,
            "data preparation": False,
            "running experiments": False,
            "results interpretation": False,
            "report writing": False,
            "report refinement": False,
        }
        
        # Create minimal workflow for demo
        max_steps = 2 if quick_mode else 5
        num_papers = 3 if quick_mode else 5
        
        try:
            print("\n🔄 Initializing Agent Laboratory workflow...")
            
            # Create research directory
            research_dir = "demo_research_output"
            os.makedirs(research_dir, exist_ok=True)
            os.makedirs(f"{research_dir}/src", exist_ok=True)
            os.makedirs(f"{research_dir}/tex", exist_ok=True)
            
            # Initialize workflow
            lab = LaboratoryWorkflow(
                research_topic=topic,
                notes=notes,
                agent_model_backbone="gpt-4o-mini",  # Use faster model for demo
                human_in_loop_flag=human_in_loop,
                openai_api_key=api_key,
                compile_pdf=False,  # Disable for demo
                num_papers_lit_review=num_papers,
                papersolver_max_steps=1,
                mlesolver_max_steps=1,
                paper_index=0,
                except_if_fail=False,
                lab_dir=research_dir
            )
            
            print("🧬 Starting research workflow...")
            print("Note: This may take several minutes to complete")
            
            # Run the workflow
            lab.perform_research()
            
            print("\n🎉 Research workflow completed successfully!")
            print(f"📁 Results saved in: {research_dir}/")
            print("\n📊 Generated files:")
            
            for root, dirs, files in os.walk(research_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, research_dir)
                    print(f"   📄 {rel_path}")
                    
        except Exception as e:
            print(f"\n❌ Error during workflow execution: {e}")
            print("This might be due to API limits, network issues, or configuration problems")
            print("Try running with --quick flag for a shorter demo")
            
    def show_features(self):
        """Display Agent Laboratory features"""
        print("\n🌟 Agent Laboratory Features")
        print("="*40)
        
        features = [
            ("🧬 Multi-Agent System", "PhD, Postdoc, Professor, ML Engineer, and Software Engineer agents collaborate"),
            ("📚 Automated Literature Review", "Searches arXiv and builds comprehensive literature understanding"),
            ("🎯 Intelligent Planning", "Agents collaborate to design and optimize research experiments"),
            ("⚙️ Code Generation", "Automatic dataset preparation and experiment implementation"),
            ("📊 Result Analysis", "Sophisticated interpretation of experimental results"),
            ("📝 Report Generation", "Publication-ready LaTeX reports with figures and citations"),
            ("🔍 Semantic Search", "Advanced paper search using sentence transformers"),
            ("🌐 Web Interface", "Beautiful, intuitive interface for managing research"),
            ("🚀 AgentRxiv Support", "Collaborative research environment for agent-generated papers"),
            ("🔧 Flexible Configuration", "Customizable workflows and model backends")
        ]
        
        for feature, description in features:
            print(f"\n{feature}")
            print(f"   {description}")

def main():
    parser = argparse.ArgumentParser(
        description="Agent Laboratory Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode",
        choices=["web", "cli", "full"],
        default="web",
        help="Demo mode to run (default: web)"
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        help="Research topic for the demo"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demo with reduced parameters"
    )
    
    parser.add_argument(
        "--features",
        action="store_true",
        help="Show Agent Laboratory features and exit"
    )
    
    args = parser.parse_args()
    
    demo = AgentLabDemo()
    demo.print_banner()
    
    if args.features:
        demo.show_features()
        return
    
    if args.mode == "web":
        demo.demo_web_interface()
    elif args.mode == "cli":
        demo.demo_cli_interface(args.topic)
    elif args.mode == "full":
        if not args.topic:
            args.topic = "Novel approaches to improve language model reasoning capabilities"
        demo.demo_full_workflow(args.topic, args.quick)

if __name__ == "__main__":
    main()