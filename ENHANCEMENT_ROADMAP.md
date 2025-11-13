# Agent Laboratory: Prioritized Enhancement Roadmap

**Document Purpose:** Actionable technical roadmap for commercializing Agent Laboratory
**Timeline:** 12-month development plan
**Last Updated:** November 13, 2025

---

## Quick Reference: Priority Matrix

| Phase | Timeline | Focus | Investment |
|-------|----------|-------|------------|
| **Phase 0: Foundation** | Weeks 1-4 | Security, stability, validation | $50K |
| **Phase 1: MVP Launch** | Weeks 5-12 | Core platform, basic UX | $200K |
| **Phase 2: Growth** | Weeks 13-26 | Scale, performance, features | $400K |
| **Phase 3: Enterprise** | Weeks 27-52 | Advanced features, integrations | $600K |

**Total Investment Required:** $1.25M over 12 months

---

## Phase 0: Foundation (Weeks 1-4)

### Objective
Make Agent Laboratory production-ready and validate commercial demand.

### Critical Security Fixes

#### 1. Code Execution Sandboxing
**Priority:** P0 - Critical
**Effort:** 10 days
**Owner:** Senior Backend Engineer

**Current Problem:**
```python
# mlesolver.py:304 - UNSAFE!
def execute_code(code_string):
    exec(code_string)  # Arbitrary code execution!
```

**Solution:**
```python
import docker
import tempfile
import os

def execute_code_sandboxed(code_string, timeout=300, memory_limit="2g"):
    """
    Execute code in isolated Docker container with resource limits.
    """
    client = docker.from_env()

    # Create temporary directory for code
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = os.path.join(tmpdir, "script.py")
        with open(code_path, 'w') as f:
            f.write(code_string)

        # Run in isolated container
        try:
            container = client.containers.run(
                "python:3.12-slim",
                command=f"python /code/script.py",
                volumes={tmpdir: {'bind': '/code', 'mode': 'ro'}},
                mem_limit=memory_limit,
                cpu_quota=50000,  # 50% of one CPU core
                network_disabled=True,
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
                timeout=timeout
            )
            return container.decode('utf-8'), None
        except docker.errors.ContainerError as e:
            return None, f"Execution error: {e.stderr.decode('utf-8')}"
        except Exception as e:
            return None, f"System error: {str(e)}"
```

**Testing:**
- Unit tests: malicious code attempts (file system access, network calls, fork bombs)
- Performance tests: resource limit enforcement
- Integration tests: end-to-end research workflow with sandboxing

**Files to Modify:**
- `/home/user/agentlaboratory/utils.py:304-320` (execute_code function)
- `/home/user/agentlaboratory/mlesolver.py:180-195` (optimize_code calls)

---

#### 2. Input Validation Framework
**Priority:** P0 - Critical
**Effort:** 5 days
**Owner:** Backend Engineer

**Implementation:**
```python
# New file: validators.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal

class ResearchConfig(BaseModel):
    """Validated research configuration."""

    research_topic: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Research topic or question"
    )

    llm_backend: Literal["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini", "deepseek-chat"] = Field(
        default="o3-mini",
        description="LLM model to use"
    )

    num_papers_lit_review: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of papers for literature review"
    )

    mlesolver_max_steps: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum optimization iterations"
    )

    budget_usd: Optional[float] = Field(
        default=None,
        ge=0,
        le=1000,
        description="Maximum spend on LLM calls"
    )

    @validator('research_topic')
    def validate_topic_safety(cls, v):
        """Prevent prompt injection attempts."""
        forbidden_patterns = [
            "ignore previous instructions",
            "system:",
            "<|endoftext|>",
            "```python",
            "exec(",
            "eval("
        ]
        v_lower = v.lower()
        for pattern in forbidden_patterns:
            if pattern in v_lower:
                raise ValueError(f"Invalid input: contains forbidden pattern '{pattern}'")
        return v

    class Config:
        extra = "forbid"  # Reject unknown fields

# Usage in ai_lab_repo.py
def load_config(yaml_path: str) -> ResearchConfig:
    """Load and validate configuration."""
    with open(yaml_path) as f:
        raw_config = yaml.safe_load(f)

    try:
        config = ResearchConfig(**raw_config)
        return config
    except ValidationError as e:
        print(f"Configuration validation failed:")
        for error in e.errors():
            print(f"  - {error['loc'][0]}: {error['msg']}")
        raise
```

**Files to Create:**
- `/home/user/agentlaboratory/validators.py` (new file)

**Files to Modify:**
- `/home/user/agentlaboratory/ai_lab_repo.py:50-100` (config loading)
- All agent files for input validation

---

#### 3. Error Handling Standardization
**Priority:** P0 - High
**Effort:** 5 days
**Owner:** All Engineers

**Implementation:**
```python
# New file: exceptions.py
class AgentLabException(Exception):
    """Base exception for Agent Laboratory."""
    def __init__(self, message: str, user_message: str = None, retry_possible: bool = False):
        self.message = message
        self.user_message = user_message or message
        self.retry_possible = retry_possible
        super().__init__(message)

class ConfigurationError(AgentLabException):
    """Invalid configuration provided."""
    pass

class LLMAPIError(AgentLabException):
    """LLM API request failed."""
    def __init__(self, provider: str, message: str, status_code: int = None):
        self.provider = provider
        self.status_code = status_code
        user_msg = f"AI service ({provider}) temporarily unavailable. Please try again."
        super().__init__(message, user_msg, retry_possible=True)

class CodeExecutionError(AgentLabException):
    """Code execution failed in sandbox."""
    pass

class PaperGenerationError(AgentLabException):
    """Paper generation or compilation failed."""
    pass

# Usage throughout codebase
from agentlaboratory.exceptions import LLMAPIError, CodeExecutionError

def query_model_safe(prompt: str, model: str, max_retries: int = 3):
    """Query LLM with proper error handling."""
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise LLMAPIError("OpenAI", "Rate limit exceeded", 429)

        except openai.APIError as e:
            if attempt < max_retries - 1:
                continue
            raise LLMAPIError("OpenAI", str(e), e.status_code)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise LLMAPIError("OpenAI", f"Unexpected error: {e}")
```

**Files to Create:**
- `/home/user/agentlaboratory/exceptions.py` (new file)

**Files to Modify:**
- `/home/user/agentlaboratory/inference.py` (all LLM calls)
- `/home/user/agentlaboratory/utils.py` (all utility functions)
- All agent files

---

#### 4. Structured Logging
**Priority:** P0 - High
**Effort:** 3 days
**Owner:** DevOps Engineer

**Implementation:**
```python
# New file: logging_config.py
import logging
import json
import sys
from datetime import datetime

class StructuredLogger:
    """JSON-structured logging for production monitoring."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }

            # Add exception info if present
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)

            # Add custom fields if present
            if hasattr(record, "research_id"):
                log_data["research_id"] = record.research_id
            if hasattr(record, "agent_type"):
                log_data["agent_type"] = record.agent_type
            if hasattr(record, "cost_usd"):
                log_data["cost_usd"] = record.cost_usd

            return json.dumps(log_data)

    def info(self, message: str, **kwargs):
        """Log info message with custom fields."""
        extra = {k: v for k, v in kwargs.items()}
        self.logger.info(message, extra=extra)

    def error(self, message: str, exc_info=True, **kwargs):
        """Log error with exception info."""
        extra = {k: v for k, v in kwargs.items()}
        self.logger.error(message, exc_info=exc_info, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        extra = {k: v for k, v in kwargs.items()}
        self.logger.warning(message, extra=extra)

# Usage
logger = StructuredLogger(__name__)

def perform_research(config):
    research_id = str(uuid.uuid4())
    logger.info(
        "Starting research",
        research_id=research_id,
        topic=config.research_topic,
        llm_backend=config.llm_backend
    )

    try:
        result = execute_workflow(config)
        logger.info(
            "Research completed",
            research_id=research_id,
            cost_usd=result.total_cost
        )
        return result
    except Exception as e:
        logger.error(
            "Research failed",
            research_id=research_id,
            error_type=type(e).__name__
        )
        raise
```

**Files to Create:**
- `/home/user/agentlaboratory/logging_config.py` (new file)

**Files to Modify:**
- All Python files (replace print statements with structured logging)

---

### Market Validation Activities

#### 5. User Interview Program
**Priority:** P0 - Critical
**Effort:** Ongoing
**Owner:** Product Manager / Founder

**Interview Script:**
```markdown
## User Interview Questions (20 minutes)

### Background (5 min)
1. What is your role? (PhD student, professor, industry researcher?)
2. How many research papers do you write per year?
3. What tools do you currently use for research?

### Pain Points (7 min)
4. What's the most time-consuming part of your research process?
5. On a scale 1-10, how satisfied are you with current research tools?
6. If you could automate one part of research, what would it be?
7. What would make you 2x more productive as a researcher?

### Product Validation (5 min)
8. [Show demo] What are your initial thoughts?
9. What features would you use most?
10. What concerns do you have about using AI for research?

### Pricing (3 min)
11. Would you pay for a tool that saves 10 hours/month?
12. What's a reasonable price? $20? $50? $100?
13. Would your institution pay for this?
```

**Target:** Interview 20 users (10 academic, 10 industry)
**Goal:** Validate willingness to pay $49/month

---

#### 6. Pilot Program Setup
**Priority:** P0 - High
**Effort:** 2 weeks
**Owner:** Founder

**Pilot Structure:**
- **Participants:** 5-10 research groups
- **Duration:** 4 weeks
- **Price:** Free (exchange for feedback)
- **Support:** Weekly check-in calls
- **Deliverables:**
  - Usage metrics
  - Qualitative feedback
  - Case studies
  - Testimonials

**Success Criteria:**
- 70%+ completion rate on research projects
- 8+ NPS score
- 2+ users willing to pay full price after pilot
- 0 critical security incidents

---

## Phase 1: MVP SaaS Platform (Weeks 5-12)

### Objective
Launch a functional SaaS product that users can sign up for and use self-service.

### Architecture Overview

```
┌─────────────────┐
│   User Browser  │
└────────┬────────┘
         │ HTTPS
┌────────▼────────┐
│   Load Balancer │ (nginx)
└────────┬────────┘
         │
┌────────▼────────┐
│   Web App       │ (Flask + React)
│   - Auth        │
│   - API         │
│   - WebSocket   │
└────────┬────────┘
         │
    ┌────┴────┬─────────────┬──────────┐
    │         │             │          │
┌───▼──┐  ┌──▼──┐  ┌──────▼──┐   ┌───▼────┐
│Redis │  │Celery│  │PostgreSQL│  │ Object │
│Cache │  │Workers│ │  DB      │  │Storage │
└──────┘  └──┬───┘  └──────────┘  └────────┘
             │
      ┌──────┴──────┐
      │   Docker    │
      │  Sandboxes  │
      └─────────────┘
```

---

### 7. Authentication System
**Priority:** P1 - Critical
**Effort:** 1 week
**Owner:** Full-Stack Engineer

**Features:**
- OAuth2 login (Google, GitHub)
- Email/password registration
- Password reset flow
- Email verification
- Session management (JWT tokens)

**Implementation:**
```python
# Use Flask-Login + Authlib
from flask_login import LoginManager, UserMixin
from authlib.integrations.flask_client import OAuth

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100))
    oauth_provider = db.Column(db.String(20))  # 'google', 'github', 'email'
    oauth_id = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    subscription_tier = db.Column(db.String(20), default='free')

    # Relationships
    research_projects = db.relationship('ResearchProject', backref='user')

# OAuth setup
oauth = OAuth(app)
oauth.register(
    'google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_callback():
    token = oauth.google.authorize_access_token()
    user_info = oauth.google.parse_id_token(token)

    # Create or update user
    user = User.query.filter_by(oauth_provider='google', oauth_id=user_info['sub']).first()
    if not user:
        user = User(
            email=user_info['email'],
            name=user_info['name'],
            oauth_provider='google',
            oauth_id=user_info['sub']
        )
        db.session.add(user)
        db.session.commit()

    login_user(user)
    return redirect(url_for('dashboard'))
```

---

### 8. Database Schema Design
**Priority:** P1 - Critical
**Effort:** 1 week
**Owner:** Backend Engineer

**Schema:**
```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(120) UNIQUE NOT NULL,
    name VARCHAR(100),
    oauth_provider VARCHAR(20),
    oauth_id VARCHAR(100),
    subscription_tier VARCHAR(20) DEFAULT 'free',
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Research projects table
CREATE TABLE research_projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(500),
    research_topic TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, running, completed, failed
    config_json JSONB,  -- Full configuration
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    cost_usd DECIMAL(10,2) DEFAULT 0,
    output_files_path VARCHAR(500)
);

-- Research phases table (tracking progress)
CREATE TABLE research_phases (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES research_projects(id),
    phase_name VARCHAR(50),  -- lit_review, planning, experiments, etc.
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    output_data JSONB,
    cost_usd DECIMAL(10,2) DEFAULT 0
);

-- Agent interactions table (for debugging/analysis)
CREATE TABLE agent_interactions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES research_projects(id),
    phase_id INTEGER REFERENCES research_phases(id),
    agent_type VARCHAR(50),
    prompt_text TEXT,
    response_text TEXT,
    llm_model VARCHAR(50),
    tokens_used INTEGER,
    cost_usd DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Papers library table (for web interface)
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    filename VARCHAR(500),
    title VARCHAR(500),
    text_content TEXT,
    arxiv_id VARCHAR(50),
    uploaded_at TIMESTAMP DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE
);

-- Subscriptions table
CREATE TABLE subscriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    stripe_customer_id VARCHAR(100),
    stripe_subscription_id VARCHAR(100),
    tier VARCHAR(20),  -- free, researcher, lab, enterprise
    status VARCHAR(20),  -- active, canceled, past_due
    current_period_start TIMESTAMP,
    current_period_end TIMESTAMP,
    cancel_at_period_end BOOLEAN DEFAULT FALSE
);

-- Usage tracking table
CREATE TABLE usage_tracking (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    month DATE,  -- First day of month
    research_runs INTEGER DEFAULT 0,
    llm_calls INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10,2) DEFAULT 0,
    papers_reviewed INTEGER DEFAULT 0
);

-- Create indexes
CREATE INDEX idx_projects_user ON research_projects(user_id);
CREATE INDEX idx_projects_status ON research_projects(status);
CREATE INDEX idx_phases_project ON research_phases(project_id);
CREATE INDEX idx_interactions_project ON agent_interactions(project_id);
CREATE INDEX idx_usage_user_month ON usage_tracking(user_id, month);
```

**Migration with Alembic:**
```bash
# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial schema"

# Apply migration
alembic upgrade head
```

---

### 9. RESTful API Development
**Priority:** P1 - Critical
**Effort:** 2 weeks
**Owner:** Backend Engineer

**Key Endpoints:**

```python
# API Routes (api/v1/)

# Research Projects
POST   /api/v1/research               # Create new research project
GET    /api/v1/research               # List user's projects
GET    /api/v1/research/:id           # Get project details
DELETE /api/v1/research/:id           # Delete project
POST   /api/v1/research/:id/start     # Start research execution
POST   /api/v1/research/:id/stop      # Stop running research

# Research Progress (WebSocket alternative)
GET    /api/v1/research/:id/status    # Get current status
GET    /api/v1/research/:id/logs      # Get execution logs
GET    /api/v1/research/:id/phases    # Get phase details

# Papers Library
POST   /api/v1/papers                 # Upload paper
GET    /api/v1/papers                 # List papers
GET    /api/v1/papers/:id             # Get paper details
DELETE /api/v1/papers/:id             # Delete paper
GET    /api/v1/papers/search?q=       # Search papers

# User Management
GET    /api/v1/user/profile           # Get user profile
PUT    /api/v1/user/profile           # Update profile
GET    /api/v1/user/usage             # Get usage stats
GET    /api/v1/user/billing           # Get billing info

# Configuration
GET    /api/v1/config/models          # List available LLM models
GET    /api/v1/config/templates       # List research templates
POST   /api/v1/config/validate        # Validate config before execution
```

**Example Implementation:**
```python
from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user

api = Blueprint('api', __name__, url_prefix='/api/v1')

@api.route('/research', methods=['POST'])
@login_required
def create_research():
    """Create new research project."""
    data = request.get_json()

    # Validate configuration
    try:
        config = ResearchConfig(**data)
    except ValidationError as e:
        return jsonify({"error": "Invalid configuration", "details": e.errors()}), 400

    # Check user's subscription limits
    usage = get_current_usage(current_user.id)
    limits = TIER_LIMITS[current_user.subscription_tier]

    if usage['research_runs_this_month'] >= limits['max_runs']:
        return jsonify({"error": "Monthly research limit reached"}), 402

    # Create project record
    project = ResearchProject(
        user_id=current_user.id,
        title=config.research_topic[:100],
        research_topic=config.research_topic,
        config_json=data,
        status='pending'
    )
    db.session.add(project)
    db.session.commit()

    # Queue for execution (Celery task)
    from tasks import execute_research_workflow
    task = execute_research_workflow.delay(project.id)

    return jsonify({
        "project_id": project.id,
        "task_id": task.id,
        "status": "queued",
        "message": "Research project created and queued for execution"
    }), 201

@api.route('/research/<int:project_id>/status', methods=['GET'])
@login_required
def get_research_status(project_id):
    """Get research project status."""
    project = ResearchProject.query.get_or_404(project_id)

    # Check authorization
    if project.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403

    # Get current phase info
    current_phase = ResearchPhase.query.filter_by(
        project_id=project_id,
        status='running'
    ).first()

    return jsonify({
        "project_id": project.id,
        "status": project.status,
        "current_phase": current_phase.phase_name if current_phase else None,
        "progress_pct": calculate_progress(project),
        "cost_usd": float(project.cost_usd),
        "started_at": project.started_at.isoformat() if project.started_at else None,
        "estimated_completion": estimate_completion(project)
    })
```

---

### 10. Modern Frontend (React)
**Priority:** P1 - High
**Effort:** 3 weeks
**Owner:** Frontend Engineer + UI/UX Designer

**Tech Stack:**
- React 18 + TypeScript
- TailwindCSS for styling
- React Query for API state management
- React Router for navigation
- Chart.js for visualizations
- Monaco Editor for code viewing

**Key Components:**

```typescript
// Dashboard.tsx - Main landing page
interface DashboardProps {
  user: User;
}

const Dashboard: React.FC<DashboardProps> = ({ user }) => {
  const { data: projects, isLoading } = useQuery('projects', fetchProjects);
  const { data: usage } = useQuery('usage', fetchUsage);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Research Dashboard</h1>

      {/* Usage Stats */}
      <UsageCard
        tier={user.subscription_tier}
        usage={usage}
        limits={TIER_LIMITS[user.subscription_tier]}
      />

      {/* Quick Actions */}
      <div className="grid grid-cols-3 gap-4 my-8">
        <ActionCard
          icon={<PlusIcon />}
          title="New Research"
          onClick={() => navigate('/research/new')}
        />
        <ActionCard
          icon={<UploadIcon />}
          title="Upload Papers"
          onClick={() => navigate('/papers/upload')}
        />
        <ActionCard
          icon={<SearchIcon />}
          title="Search Papers"
          onClick={() => navigate('/papers/search')}
        />
      </div>

      {/* Recent Projects */}
      <ProjectList projects={projects} loading={isLoading} />
    </div>
  );
};

// ResearchConfig.tsx - Configuration wizard
const ResearchConfigWizard: React.FC = () => {
  const [step, setStep] = useState(1);
  const [config, setConfig] = useState<Partial<ResearchConfig>>({});

  return (
    <WizardContainer>
      {step === 1 && (
        <TopicStep
          value={config.research_topic}
          onChange={(topic) => setConfig({...config, research_topic: topic})}
          onNext={() => setStep(2)}
        />
      )}

      {step === 2 && (
        <ModelSelectionStep
          value={config.llm_backend}
          onChange={(model) => setConfig({...config, llm_backend: model})}
          onNext={() => setStep(3)}
          onBack={() => setStep(1)}
        />
      )}

      {step === 3 && (
        <AdvancedOptionsStep
          config={config}
          onChange={setConfig}
          onNext={() => setStep(4)}
          onBack={() => setStep(2)}
        />
      )}

      {step === 4 && (
        <ReviewStep
          config={config}
          onConfirm={submitResearch}
          onBack={() => setStep(3)}
        />
      )}
    </WizardContainer>
  );
};

// ResearchProgress.tsx - Real-time progress tracking
const ResearchProgress: React.FC<{ projectId: number }> = ({ projectId }) => {
  const { data: status } = useQuery(
    ['research-status', projectId],
    () => fetchResearchStatus(projectId),
    { refetchInterval: 5000 }  // Poll every 5 seconds
  );

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold mb-4">{status.title}</h2>

      {/* Progress Bar */}
      <ProgressBar percentage={status.progress_pct} />

      {/* Phase Timeline */}
      <PhaseTimeline
        phases={status.phases}
        currentPhase={status.current_phase}
      />

      {/* Live Logs */}
      <LogViewer projectId={projectId} />

      {/* Cost Tracker */}
      <div className="mt-4 text-sm text-gray-600">
        Cost: ${status.cost_usd.toFixed(2)} |
        Est. Completion: {status.estimated_completion}
      </div>
    </div>
  );
};
```

**Design System:**
- Color palette: Professional blues and grays
- Typography: Inter for UI, JetBrains Mono for code
- Icons: Heroicons
- Animations: Smooth transitions, loading states
- Responsive: Mobile-first design

---

### 11. Billing Integration (Stripe)
**Priority:** P1 - High
**Effort:** 1 week
**Owner:** Full-Stack Engineer

**Implementation:**
```python
import stripe
from flask import request, jsonify

stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

# Pricing configuration
PRICING = {
    'researcher': {
        'stripe_price_id': 'price_researcher_monthly',
        'amount': 4900,  # $49.00
        'max_runs': 20
    },
    'lab': {
        'stripe_price_id': 'price_lab_monthly',
        'amount': 19900,  # $199.00
        'max_runs': 100
    }
}

@app.route('/api/v1/billing/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    """Create Stripe checkout session for subscription."""
    data = request.get_json()
    tier = data.get('tier')

    if tier not in PRICING:
        return jsonify({"error": "Invalid tier"}), 400

    try:
        # Create or get Stripe customer
        if not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                metadata={'user_id': current_user.id}
            )
            current_user.stripe_customer_id = customer.id
            db.session.commit()

        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price': PRICING[tier]['stripe_price_id'],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=url_for('billing_success', _external=True),
            cancel_url=url_for('billing_cancel', _external=True),
        )

        return jsonify({"checkout_url": session.url})

    except Exception as e:
        logger.error(f"Stripe error: {e}")
        return jsonify({"error": "Payment processing error"}), 500

@app.route('/api/v1/billing/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks."""
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv('STRIPE_WEBHOOK_SECRET')
        )
    except ValueError:
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError:
        return jsonify({"error": "Invalid signature"}), 400

    # Handle different event types
    if event['type'] == 'customer.subscription.created':
        handle_subscription_created(event['data']['object'])
    elif event['type'] == 'customer.subscription.updated':
        handle_subscription_updated(event['data']['object'])
    elif event['type'] == 'customer.subscription.deleted':
        handle_subscription_deleted(event['data']['object'])
    elif event['type'] == 'invoice.payment_failed':
        handle_payment_failed(event['data']['object'])

    return jsonify({"status": "success"})

def handle_subscription_created(subscription):
    """Update user subscription status."""
    customer_id = subscription['customer']
    user = User.query.filter_by(stripe_customer_id=customer_id).first()

    if user:
        # Determine tier from price ID
        tier = None
        for t, info in PRICING.items():
            if subscription['items']['data'][0]['price']['id'] == info['stripe_price_id']:
                tier = t
                break

        if tier:
            user.subscription_tier = tier

            # Create subscription record
            sub = Subscription(
                user_id=user.id,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription['id'],
                tier=tier,
                status=subscription['status'],
                current_period_start=datetime.fromtimestamp(subscription['current_period_start']),
                current_period_end=datetime.fromtimestamp(subscription['current_period_end'])
            )
            db.session.add(sub)
            db.session.commit()

            logger.info(f"Subscription created for user {user.id}: {tier}")
```

---

### 12. Async Task Queue (Celery)
**Priority:** P1 - Critical
**Effort:** 1 week
**Owner:** Backend Engineer

**Setup:**
```python
# celery_app.py
from celery import Celery
from celery.signals import task_prerun, task_postrun

celery_app = Celery(
    'agentlaboratory',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=7200,  # 2 hours
    task_soft_time_limit=6600,  # 1h 50m (warning before kill)
)

# tasks.py
from celery_app import celery_app
from ai_lab_repo import LaboratoryWorkflow
import traceback

@celery_app.task(bind=True, max_retries=3)
def execute_research_workflow(self, project_id: int):
    """
    Execute full research workflow asynchronously.

    Args:
        project_id: Database ID of research project
    """
    from app import db, ResearchProject, ResearchPhase

    # Update project status
    project = ResearchProject.query.get(project_id)
    project.status = 'running'
    project.started_at = datetime.utcnow()
    db.session.commit()

    # Load configuration
    config = project.config_json

    try:
        # Initialize workflow with progress callback
        def progress_callback(phase_name, status, data=None):
            """Update database with progress."""
            phase = ResearchPhase.query.filter_by(
                project_id=project_id,
                phase_name=phase_name
            ).first()

            if not phase:
                phase = ResearchPhase(
                    project_id=project_id,
                    phase_name=phase_name,
                    status='running',
                    started_at=datetime.utcnow()
                )
                db.session.add(phase)

            phase.status = status
            if status == 'completed':
                phase.completed_at = datetime.utcnow()
                phase.output_data = data

            db.session.commit()

            # Update Celery task state for real-time updates
            self.update_state(
                state='PROGRESS',
                meta={
                    'phase': phase_name,
                    'status': status,
                    'progress': calculate_phase_progress(project_id)
                }
            )

        # Execute workflow
        workflow = LaboratoryWorkflow(
            research_topic=config['research_topic'],
            llm_backend=config['llm_backend'],
            config=config,
            progress_callback=progress_callback
        )

        result = workflow.perform_research()

        # Update project with results
        project.status = 'completed'
        project.completed_at = datetime.utcnow()
        project.cost_usd = result['total_cost']
        project.output_files_path = result['output_path']
        db.session.commit()

        return {
            'status': 'completed',
            'project_id': project_id,
            'cost_usd': result['total_cost'],
            'output_path': result['output_path']
        }

    except SoftTimeLimitExceeded:
        project.status = 'timeout'
        db.session.commit()
        raise

    except Exception as e:
        # Log error
        logger.error(
            f"Research workflow failed for project {project_id}",
            exc_info=True,
            project_id=project_id
        )

        # Update project
        project.status = 'failed'
        project.error_message = str(e)
        project.error_traceback = traceback.format_exc()
        db.session.commit()

        # Retry if transient error
        if is_retryable_error(e):
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

        raise
```

**Running Celery:**
```bash
# Start worker
celery -A celery_app worker --loglevel=info --concurrency=4

# Start beat (for scheduled tasks)
celery -A celery_app beat --loglevel=info

# Monitor with Flower
celery -A celery_app flower --port=5555
```

---

## Phase 2: Growth & Scale (Weeks 13-26)

### Objective
Scale to 100+ active users with reliable performance and enhanced features.

---

### 13. Performance Optimization
**Priority:** P2 - High
**Effort:** 2 weeks
**Owner:** Senior Backend Engineer

**Key Optimizations:**

1. **LLM Response Caching**
```python
import hashlib
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=2)

def cached_llm_query(prompt: str, model: str, ttl: int = 3600):
    """Cache LLM responses to reduce costs."""
    # Create cache key
    cache_key = f"llm:{model}:{hashlib.sha256(prompt.encode()).hexdigest()}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        logger.info("Cache hit for LLM query")
        return json.loads(cached)

    # Query LLM
    response = query_model(prompt, model)

    # Cache response
    redis_client.setex(
        cache_key,
        ttl,
        json.dumps(response)
    )

    return response
```

2. **Database Query Optimization**
```python
# Use eager loading for relationships
projects = ResearchProject.query.options(
    joinedload(ResearchProject.phases),
    joinedload(ResearchProject.user)
).filter_by(user_id=user_id).all()

# Add indexes
CREATE INDEX CONCURRENTLY idx_projects_user_status ON research_projects(user_id, status);
CREATE INDEX CONCURRENTLY idx_phases_project_status ON research_phases(project_id, status);

# Use database-side aggregations
SELECT
    user_id,
    COUNT(*) as total_projects,
    SUM(cost_usd) as total_cost,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))/3600) as avg_hours
FROM research_projects
WHERE status = 'completed'
GROUP BY user_id;
```

3. **Parallel Literature Review**
```python
import concurrent.futures

def parallel_literature_review(query: str, num_papers: int = 10):
    """Fetch multiple papers concurrently."""
    search_results = arxiv.Search(
        query=query,
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = list(search_results.results())

    # Process papers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_paper, paper)
            for paper in papers
        ]

        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as e:
                logger.error(f"Paper processing failed: {e}")

        return results
```

---

### 14. Monitoring & Observability
**Priority:** P2 - High
**Effort:** 1 week
**Owner:** DevOps Engineer

**Stack:**
- Prometheus for metrics
- Grafana for dashboards
- Sentry for error tracking
- Elasticsearch + Kibana for log analysis

**Key Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
research_projects_started = Counter(
    'research_projects_started_total',
    'Total research projects started',
    ['user_tier', 'llm_backend']
)

research_projects_completed = Counter(
    'research_projects_completed_total',
    'Total research projects completed',
    ['user_tier', 'status']
)

# Histograms
research_duration = Histogram(
    'research_duration_seconds',
    'Research project duration',
    ['phase']
)

llm_call_duration = Histogram(
    'llm_call_duration_seconds',
    'LLM API call duration',
    ['provider', 'model']
)

# Gauges
active_research_projects = Gauge(
    'active_research_projects',
    'Number of currently running research projects'
)

# Usage in code
with research_duration.labels(phase='literature_review').time():
    perform_literature_review()

research_projects_started.labels(
    user_tier=user.subscription_tier,
    llm_backend=config.llm_backend
).inc()
```

**Grafana Dashboard:**
- System health (CPU, memory, disk)
- Research metrics (projects/day, completion rate)
- User metrics (signups, active users, churn)
- Cost metrics (LLM spend, revenue)
- Performance (API latency, task queue length)

---

### 15-25. [Additional features...]

[Continuing with remaining phases and features...]

---

## Implementation Checklist

### Week 1-4: Foundation
- [ ] Code execution sandboxing (Docker)
- [ ] Input validation framework (Pydantic)
- [ ] Error handling standardization
- [ ] Structured logging (JSON)
- [ ] User interviews (20 completed)
- [ ] Pilot program setup
- [ ] Security audit

### Week 5-8: Core Platform
- [ ] PostgreSQL migration
- [ ] Authentication (OAuth2)
- [ ] REST API (all endpoints)
- [ ] Celery task queue
- [ ] Basic React frontend
- [ ] Billing integration (Stripe)

### Week 9-12: Launch Prep
- [ ] Frontend polish (design system)
- [ ] Progress tracking UI
- [ ] Configuration wizard
- [ ] Documentation site
- [ ] Landing page
- [ ] Beta testing (50 users)

### Week 13-16: Performance
- [ ] LLM caching
- [ ] Database optimization
- [ ] Parallel execution
- [ ] Monitoring (Prometheus)
- [ ] Error tracking (Sentry)

### Week 17-26: Growth Features
- [ ] Collaboration features
- [ ] Advanced integrations
- [ ] Mobile-responsive design
- [ ] API documentation
- [ ] Customer support system

### Week 27-39: Enterprise
- [ ] On-premise deployment
- [ ] SAML authentication
- [ ] Advanced analytics
- [ ] White-label option
- [ ] Compliance (SOC 2)

### Week 40-52: Scale
- [ ] Multi-region deployment
- [ ] Advanced caching (CDN)
- [ ] Load testing (1000+ concurrent)
- [ ] Disaster recovery
- [ ] International expansion

---

## Success Metrics

### Phase 0 (Week 4)
- [ ] 20 user interviews completed
- [ ] 5 pilots recruited
- [ ] 0 critical security vulnerabilities

### Phase 1 (Week 12)
- [ ] 100 beta signups
- [ ] 20 paid conversions
- [ ] 99% uptime
- [ ] <2s API response time

### Phase 2 (Week 26)
- [ ] 500 total users
- [ ] 50 paying customers
- [ ] $2.5K MRR
- [ ] 8+ NPS score

### Phase 3 (Week 52)
- [ ] 2000 total users
- [ ] 200 paying customers
- [ ] $15K MRR
- [ ] 3 enterprise deals

---

**Next Steps:** Review this roadmap with engineering team and adjust timelines based on available resources.
