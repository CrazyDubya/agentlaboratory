import random, time

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from flask_sqlalchemy import SQLAlchemy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///papers.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Paper(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    text = db.Column(db.Text, nullable=True)

def update_papers_from_uploads():
    for _tries in range(5):
        try:
            uploads_dir = app.config['UPLOAD_FOLDER']
            file_list = os.listdir(uploads_dir)
            print("Files in uploads folder:", file_list)
            for filename in file_list:
                if filename.lower().endswith('.pdf'):
                    # Check if file is already in the DB
                    if not Paper.query.filter_by(filename=filename).first():
                        print("Processing file:", filename)
                        file_path = os.path.join(uploads_dir, filename)
                        extracted_text = ""
                        try:
                            reader = PdfReader(file_path)
                            for page in reader.pages:
                                text = page.extract_text()
                                if text:
                                    extracted_text += text
                        except Exception as e:
                            flash(f'Error processing {filename}: {e}')
                            continue
                        if not extracted_text.strip():
                            print(f"Warning: No text extracted from {filename}")
                        else:
                            print(f"Extracted {len(extracted_text)} characters from {filename}")
                        new_paper = Paper(filename=filename, text=extracted_text)
                        db.session.add(new_paper)
            db.session.commit()
            return
        except Exception as e:
            print("WEB SERVER LOAD EXCEPTION", e, str(e))
            time.sleep(random.randint(5, 15))
    return
    #raise Exception("FAILED TO UPDATE")

# Load a pre-trained sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model_loaded = True
except Exception as e:
    print(f"⚠️ Could not load sentence transformer model: {e}")
    print("   Semantic search will be disabled in offline mode")
    model = None
    model_loaded = False

SEARCH_ENABLED = model_loaded
if SEARCH_ENABLED:
    print("✅ Sentence transformer model loaded successfully")
@app.route('/update', methods=['GET'])
def update_on_demand():
    update_papers_from_uploads()
    return jsonify({"message": "Uploads folder processed successfully."})

@app.route('/')
def index():
    update_papers_from_uploads()
    papers = Paper.query.all()
    return render_template('index.html', papers=papers)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'pdf' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['pdf']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            extracted_text = ""
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text
            except Exception as e:
                flash(f'Error processing PDF: {e}')
            new_paper = Paper(filename=filename, text=extracted_text)
            db.session.add(new_paper)
            db.session.commit()
            flash('File uploaded and processed successfully!')
            return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if query:
        if not SEARCH_ENABLED:
            # Fallback: simple text matching when semantic search is not available
            papers = Paper.query.all()
            papers_with_scores = []
            query_lower = query.lower()
            
            for paper in papers:
                if paper.text:
                    # Simple text matching score
                    text_lower = paper.text.lower()
                    # Count occurrences of query words
                    query_words = query_lower.split()
                    matches = sum(text_lower.count(word) for word in query_words)
                    if matches > 0:
                        # Normalize score between 0 and 1
                        score = min(matches * 0.1, 1.0)
                        papers_with_scores.append((paper, score))
            
            papers_sorted = sorted(papers_with_scores, key=lambda x: x[1], reverse=True)
            
        else:
            # Use semantic search when available
            papers = Paper.query.all()
            query_embedding = model.encode([query])
            paper_texts = [paper.text for paper in papers if paper.text]
            if not paper_texts:
                return render_template('search.html', papers=[], query=query)
            paper_embeddings = model.encode(paper_texts)
            similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
            papers_with_scores = list(zip([p for p in papers if p.text], similarities))
            papers_sorted = sorted(papers_with_scores, key=lambda x: x[1], reverse=True)
            
        return render_template('search.html', papers=papers_sorted, query=query, search_enabled=SEARCH_ENABLED)
    return render_template('search.html', papers=[], query=query, search_enabled=SEARCH_ENABLED)

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    papers = Paper.query.all()
    if not papers:
        return jsonify({'query': query, 'results': []})
        
    if not SEARCH_ENABLED:
        # Fallback: simple text matching
        papers_with_scores = []
        query_lower = query.lower()
        
        for paper in papers:
            if paper.text:
                text_lower = paper.text.lower()
                query_words = query_lower.split()
                matches = sum(text_lower.count(word) for word in query_words)
                if matches > 0:
                    score = min(matches * 0.1, 1.0)
                    papers_with_scores.append((paper, score))
        
        papers_sorted = sorted(papers_with_scores, key=lambda x: x[1], reverse=True)
        
    else:
        # Use semantic search when available
        query_embedding = model.encode([query])
        paper_texts = [paper.text for paper in papers if paper.text]
        if not paper_texts:
            return jsonify({'query': query, 'results': []})
        paper_embeddings = model.encode(paper_texts)
        similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
        papers_with_scores = list(zip([p for p in papers if p.text], similarities))
        papers_sorted = sorted(papers_with_scores, key=lambda x: x[1], reverse=True)
        
    results = []
    for paper, score in papers_sorted:
        pdf_url = url_for('uploaded_file', filename=paper.filename, _external=True)
        results.append({
            'id': paper.id,
            'filename': paper.filename,
            'similarity': float(score),
            'pdf_url': pdf_url
        })
    return jsonify({'query': query, 'results': results, 'search_mode': 'semantic' if SEARCH_ENABLED else 'text_matching'})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='application/pdf')

@app.route('/view/<int:paper_id>')
def view_pdf(paper_id):
    paper = Paper.query.get_or_404(paper_id)
    pdf_url = url_for('uploaded_file', filename=paper.filename, _external=True)
    return render_template('view.html', paper=paper, pdf_url=pdf_url)

@app.route('/demo')
def demo():
    return render_template('demo.html')


def run_app(port=5000):
    # Reset the database by removing the existing file
    db_path = "papers.db"
    if os.path.exists("instance/" + db_path):
        os.remove("instance/" + db_path)
    with app.app_context():
        db.create_all()
        # Create sample data after database is initialized
        create_sample_data_in_db()
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=False, port=port)

def create_sample_data_in_db():
    """Create sample research papers in the database"""
    sample_papers = [
        {
            "filename": "chain_of_thought_reasoning.txt",
            "content": """Title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

Abstract: This paper introduces chain-of-thought prompting, a simple method that enables large language models to generate intermediate reasoning steps when solving complex problems. We show that chain-of-thought prompting significantly improves performance on arithmetic reasoning tasks, common sense reasoning, and symbolic reasoning tasks.

Introduction: Large language models have shown remarkable capabilities in many natural language processing tasks. However, they often struggle with tasks that require complex reasoning and multi-step problem solving. Chain-of-thought prompting addresses this limitation by encouraging the model to work through problems step-by-step.

Methods: We evaluate chain-of-thought prompting on several reasoning benchmarks including GSM8K (grade school math problems), MATH (mathematical competition problems), and CommonsenseQA. The method involves providing examples that show the reasoning process explicitly.

Results: Our experiments show that chain-of-thought prompting leads to substantial improvements across all tested domains. On GSM8K, accuracy improved from 17.9% to 58.1% with PaLM 540B.

Conclusion: Chain-of-thought prompting is a simple yet effective technique for improving the reasoning capabilities of large language models. This approach could be valuable for many applications requiring logical reasoning."""
        },
        {
            "filename": "mathematical_reasoning_llms.txt", 
            "content": """Title: Mathematical Reasoning in Large Language Models

Abstract: We investigate the mathematical reasoning capabilities of large language models through comprehensive evaluation on various mathematical tasks. Our study reveals both strengths and limitations of current models in mathematical problem solving.

Introduction: Mathematical reasoning is a fundamental cognitive skill that involves logical thinking, pattern recognition, and systematic problem solving. Recent advances in large language models have shown promising results in mathematical tasks, but systematic evaluation is needed.

Dataset: We use the MATH dataset containing 12,500 challenging mathematical problems spanning algebra, geometry, number theory, and other areas. Problems are graded at high school competition level.

Methodology: We evaluate several state-of-the-art language models including GPT-3, GPT-4, and PaLM using various prompting strategies. We measure both final answer accuracy and reasoning quality.

Results: GPT-4 achieves 42.5% accuracy on MATH, while GPT-3 achieves 23.5%. Chain-of-thought prompting consistently improves performance across all models. Error analysis reveals common failure modes."""
        },
        {
            "filename": "agent_laboratory_research.txt",
            "content": """Title: Agent Laboratory: Using LLM Agents as Research Assistants

Abstract: We introduce Agent Laboratory, a novel framework that leverages multiple specialized AI agents to automate the research process. Our system demonstrates how collaborative AI agents can conduct literature reviews, formulate hypotheses, design experiments, and generate research reports.

Introduction: The research process traditionally requires significant human expertise and time investment. Agent Laboratory addresses this challenge by coordinating multiple AI agents with specialized roles: PhD Student, Postdoc, Professor, ML Engineer, and Software Engineer agents.

Methodology: Our multi-agent system operates through a structured workflow: literature review via arXiv integration, collaborative planning between agents, automated experiment design and implementation, real-time result analysis, and publication-ready report generation.

Results: Agent Laboratory successfully demonstrates state-of-the-art performance on mathematical reasoning benchmarks, achieving 70.2% accuracy on MATH dataset through novel prompting techniques developed autonomously by the agent team.

Conclusion: This work opens new possibilities for AI-assisted research, showing how specialized agents can collaborate effectively to advance scientific knowledge."""
        }
    ]
    
    # Clear existing sample papers
    try:
        db.session.query(Paper).filter(Paper.filename.like('%chain_of_thought%')).delete()
        db.session.query(Paper).filter(Paper.filename.like('%mathematical_reasoning%')).delete()
        db.session.query(Paper).filter(Paper.filename.like('%agent_laboratory%')).delete()
        db.session.commit()
    except Exception:
        pass  # Table might not exist yet
    
    # Add sample papers to database
    try:
        for paper_data in sample_papers:
            paper = Paper(
                filename=paper_data["filename"],
                text=paper_data["content"]
            )
            db.session.add(paper)
        db.session.commit()
        print("✅ Added sample papers to database")
    except Exception as e:
        print(f"⚠️ Could not add sample papers to database: {e}")

if __name__ == '__main__':
    run_app()