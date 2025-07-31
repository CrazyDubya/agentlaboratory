# 🧬 Agent Laboratory Demo Documentation

## Overview

This demo showcases the full functionality of Agent Laboratory - a cutting-edge research framework that uses collaborative AI agents to automate the entire research process from literature review to publication-ready reports.

## 🚀 Quick Start

### Option 1: Web Interface Demo (Recommended)
```bash
python simple_demo.py --mode web
```
Then visit: http://localhost:5000

### Option 2: CLI Demo
```bash
python simple_demo.py --mode cli --topic "your research topic"
```

### Option 3: Full Research Workflow (Requires API Keys)
```bash
export OPENAI_API_KEY="your-key-here"
python ai_lab_repo.py --yaml-location "experiment_configs/MATH_agentlab.yaml"
```

## 🌟 Demo Features

### 1. Interactive Web Interface
- **Beautiful, responsive design** with modern UI/UX
- **Paper library management** with automatic content extraction
- **Semantic search** (online) or text matching (offline mode)
- **Interactive research workflow demo** with real-time simulations

### 2. Multi-Agent Research System
- **PhD Student Agent** - Research execution and implementation
- **Postdoc Agent** - Strategic planning and methodology design
- **Professor Agent** - Oversight, guidance, and quality assurance
- **ML Engineer Agent** - Technical implementation and optimization
- **Software Engineer Agent** - Code development and system integration

### 3. Complete Research Pipeline
1. **Literature Review** - Automated paper search and analysis
2. **Plan Formulation** - Collaborative research design
3. **Data Preparation** - Dataset processing and experimental setup
4. **Running Experiments** - Parallelized execution and monitoring
5. **Results Analysis** - Statistical analysis and interpretation
6. **Report Writing** - Publication-ready LaTeX documents

## 🎯 Demo Scenarios

### MATH Benchmark Research
Demonstrates how Agent Laboratory can:
- Develop novel prompting techniques for mathematical reasoning
- Achieve state-of-the-art performance on MATH-500 dataset
- Generate comprehensive experimental analysis
- Produce publication-ready research reports

### Custom Research Topics
Users can define their own research questions and watch agents:
- Conduct autonomous literature reviews
- Formulate research hypotheses
- Design and execute experiments
- Analyze results and generate reports

## 🔧 Configuration Options

### Agent Models
- **o3-mini** (Recommended) - Fast and efficient
- **GPT-4o** - High-quality reasoning
- **DeepSeek v3** - Cost-effective alternative

### Research Depth
- **3 Papers (Fast)** - Quick demonstration
- **5 Papers (Balanced)** - Standard research depth
- **7 Papers (Comprehensive)** - Thorough analysis

### Execution Modes
- **Autonomous** - Fully automated workflow
- **Interactive** - Human-in-the-loop validation

## 🏗️ Technical Architecture

### Web Framework
- **Flask** - Lightweight web framework
- **SQLAlchemy** - Database ORM for paper management
- **Jinja2** - Template engine for dynamic content

### AI/ML Components
- **Sentence Transformers** - Semantic search capabilities
- **OpenAI API** - LLM backend for agent reasoning
- **PyPDF2** - PDF text extraction and processing

### Research Tools
- **arXiv Integration** - Automated paper discovery
- **LaTeX Generation** - Publication-ready formatting
- **Statistical Analysis** - Comprehensive result evaluation

## 📊 Demo Statistics

### Performance Metrics
- **Processing Speed** - Handles large document collections efficiently
- **Search Accuracy** - High-quality semantic matching
- **User Experience** - Intuitive interface with real-time feedback

### Supported Formats
- **PDF Papers** - Automatic text extraction
- **Research Topics** - Natural language input
- **Export Formats** - LaTeX, PDF, Markdown

## 🌐 Web Interface Features

### Paper Library
- Upload and process PDF research papers
- Automatic text extraction and indexing
- Browse papers with processing status indicators
- View extracted content with clean formatting

### Search Functionality
- Semantic search using transformer models
- Fallback text matching for offline mode
- Relevance scoring and result ranking
- Interactive search suggestions

### Research Demo
- Two pre-configured research scenarios
- Custom research topic input
- Real-time workflow simulation
- Configuration options for different use cases

## 💡 Best Practices

### For Optimal Performance
1. **Use high-quality PDF papers** with clear text (not scanned images)
2. **Provide specific research topics** for better agent collaboration
3. **Configure appropriate model backends** based on computational resources
4. **Monitor processing status** for large document collections

### For Production Use
1. **Set up proper API keys** for full functionality
2. **Configure LaTeX environment** for report generation
3. **Allocate sufficient computational resources** for parallel processing
4. **Implement proper error handling** and logging

## 🚀 Advanced Features

### AgentRxiv Integration
- Collaborative research platform for AI-generated papers
- Cross-agent knowledge sharing and building
- Cumulative research progress tracking

### Parallel Processing
- Multi-threaded experiment execution
- Scalable to multiple research labs
- Efficient resource utilization

### Customization
- Flexible YAML configuration
- Multiple LLM backend support
- Configurable workflow parameters

## 🔍 Troubleshooting

### Common Issues
- **Semantic search disabled**: Install sentence-transformers with internet access
- **API errors**: Verify OpenAI/DeepSeek API keys are properly set
- **PDF processing fails**: Ensure PDFs contain extractable text
- **Slow performance**: Reduce paper count or use faster models

### System Requirements
- **Python 3.8+** with required dependencies
- **4GB+ RAM** for optimal performance
- **Internet connection** for full semantic search
- **API access** for complete research workflows

## 📈 Future Enhancements

### Planned Features
- **Multi-language support** for international research
- **Advanced visualization** for research insights
- **Integration with more databases** beyond arXiv
- **Real-time collaboration** between multiple users

### Research Applications
- **Academic research acceleration**
- **Literature review automation** 
- **Hypothesis generation and testing**
- **Meta-analysis and systematic reviews**

## 📞 Support

For questions, issues, or contributions:
- Check the repository documentation
- Review existing issues and discussions
- Submit detailed bug reports with logs
- Contribute improvements via pull requests

---

**Agent Laboratory** - Revolutionizing research through AI collaboration 🧬