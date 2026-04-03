# Phase 5 — Bedrock Agents & AWS Deployment

**Git tag:** `v1.0.0`  
**Deliverable:** Oak Assist + Fraud Agent running on AWS Bedrock; full stack deployable via `cdk deploy`.

**Meetup demo:** File a claim via Oak Assist, watch the fraud agent flag a suspicious record and explain why, show CDK deploy for attendees pushing to their own AWS account.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Component Swap: Local → AWS](#2-component-swap-local--aws)
3. [Bedrock Agent Definitions](#3-bedrock-agent-definitions)
4. [CDK Stacks](#4-cdk-stacks)
5. [Deploy Commands](#5-deploy-commands)
6. [Intelligent Prompt Routing (Cost Optimization)](#6-intelligent-prompt-routing-cost-optimization)
7. [Verification & Git Tag](#7-verification--git-tag)

---

## 1. Prerequisites

```bash
# AWS CLI — Linux / macOS
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install
```

```powershell
# Windows — download and run the MSI installer
# https://awscli.amazonaws.com/AWSCLIV2.msi
```

```bash
# Configure credentials
aws configure   # enter Access Key, Secret Key, region (us-east-1), output (json)

# Node.js v20 LTS required for CDK CLI
npm install -g aws-cdk
cdk --version

# Bootstrap CDK — one-time per account/region
cdk bootstrap aws://ACCOUNT_ID/us-east-1
```

---

## 2. Component Swap: Local → AWS

Every component has a direct swap. No code rewrites — environment variables and mode flags control which backend is used.

| Component | Local (Phases 1–4) | AWS (Phase 5) | Change Required |
|-----------|-------------------|---------------|-----------------|
| PostgreSQL | Docker (`pgvector/pgvector:pg16`) | RDS PostgreSQL 16 + pgvector extension | Update `DB_HOST` in `.env` |
| Vector store | pgvector on Docker | pgvector on RDS (same extension) | Same queries, new host |
| Embedding | `sentence-transformers` local | Titan Embeddings V2 via Bedrock | Set `mode=bedrock` in `embed_and_load.py` |
| Generation | Ollama (Llama 3.1 8B) | Claude Haiku / Sonnet via Bedrock | Set `mode=bedrock` in `rag_pipeline.py` |
| FastAPI | `uvicorn` local | Lambda + API Gateway via `mangum` | `handler = Mangum(app)` already in place |
| ML models | Local `.json` model files | Lambda with model file in deployment package | Add model file to Lambda zip |

> **Critical flag:** Verify all model ID strings in `.env` match current Bedrock IDs before deploying. Using deprecated IDs (e.g. old Claude 3.5 Sonnet strings) instead of `claude-sonnet-4-6` can double inference costs. Audit every reference — `.env`, CDK stacks, and any hardcoded strings in agent files.

---

## 3. Bedrock Agent Definitions

### Oak Assist / FNOL Agent (`ai/agents/claims_agent/agent.py`)

```python
"""
claims_agent/agent.py — Oak Assist FNOL intake agent.
Handles: new claim filing, status checks, billing disputes, coverage questions.
Model: claude-sonnet-4-6 (multi-turn conversation)
"""
import boto3, json, os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are Oak Assist, the AI claims intake agent for Avvaru Iron Oak Insurance.
Your role:
  1. Help customers file a First Notice of Loss (FNOL) for vehicle incidents.
  2. Answer coverage questions grounded in the customer's policy documents.
  3. Escalate complex disputes to a human adjuster.

Always:
  - Ask for policy number and incident date before proceeding with a new claim.
  - Cite whether your answer comes from the customer's policy or general FAQ.
  - Do not fabricate coverage details — if you don't have the information, say so.
  - Follow state-specific rules (no-fault, PIP requirements) based on the customer's state.

Tone: professional, empathetic, concise.
"""

# TODO: prompt injection hardening — validate user_message before passing to agent.
# Strip or escape imperative phrases directed at the model.
# See CROSS_PHASE.md §9.1.

# TODO: pass guardrailIdentifier and guardrailVersion to invoke_agent call.
# See CROSS_PHASE.md §9.2.

def invoke_agent(session_id: str, user_message: str) -> str:
    """Single turn in a multi-turn FNOL session."""
    client = boto3.client("bedrock-agent-runtime", region_name=os.getenv("AWS_DEFAULT_REGION","us-east-1"))
    resp = client.invoke_agent(
        agentId=os.getenv("BEDROCK_CLAIMS_AGENT_ID"),
        agentAliasId=os.getenv("BEDROCK_CLAIMS_AGENT_ALIAS_ID"),
        sessionId=session_id,
        inputText=user_message,
    )
    completion = ""
    for event in resp["completion"]:
        if "chunk" in event:
            completion += event["chunk"]["bytes"].decode("utf-8")
    return completion
```

### Fraud Agent (`ai/agents/fraud_agent/agent.py`)

```python
"""
fraud_agent/agent.py — explains suspicious claims and recommends investigation actions.
Model: claude-sonnet-4-6 (reasoning + explanation)
Triggered by: fraud_detection model returning fraud_score >= 0.5
"""
import boto3, json, os

SYSTEM_PROMPT = """
You are the AIOI Fraud Analysis Agent. You receive a claim record flagged by the XGBoost
fraud classifier with a fraud score and a list of fraud signals.

Your task:
  1. Explain in plain English why this claim was flagged.
  2. List the specific signals and what each one suggests.
  3. Recommend next investigation steps (SIU referral, telematics audit, customer interview).
  4. Assign a risk tier: LOW / MEDIUM / HIGH based on signal combination.

Be specific and cite the data provided — do not generalize.
"""

def analyze_claim(claim: dict, fraud_result: dict) -> str:
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION","us-east-1"))
    prompt = (
        f"Claim data:\n{json.dumps(claim, indent=2)}\n\n"
        f"Fraud detection result:\n{json.dumps(fraud_result, indent=2)}\n\n"
        "Provide your fraud analysis."
    )
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = client.invoke_model(
        modelId=os.getenv("BEDROCK_MODEL_ID_SONNET"),
        body=body,
        contentType="application/json",
    )
    return json.loads(resp["body"].read())["content"][0]["text"]
```

---

## 4. CDK Stacks

### `infra/cdk/app.py`

```python
#!/usr/bin/env python3
import aws_cdk as cdk
from stacks.database_stack import DatabaseStack
from stacks.api_stack import ApiStack
from stacks.bedrock_stack import BedrockStack

app = cdk.App()
env = cdk.Environment(account=app.node.try_get_context("account"),
                      region=app.node.try_get_context("region") or "us-east-1")

db_stack      = DatabaseStack(app, "AIOI-Database", env=env)
api_stack     = ApiStack(app, "AIOI-Api", vpc=db_stack.vpc, env=env)
bedrock_stack = BedrockStack(app, "AIOI-Bedrock", env=env)

app.synth()
```

### `infra/cdk/stacks/database_stack.py`

```python
from aws_cdk import Stack, aws_rds as rds, aws_ec2 as ec2
from constructs import Construct

class DatabaseStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        self.vpc = ec2.Vpc(self, "AIOIVpc", max_azs=2, nat_gateways=1)

        self.db = rds.DatabaseInstance(self, "AIOIPostgres",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_16
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM
            ),
            vpc=self.vpc,
            database_name="aioi",
            credentials=rds.Credentials.from_generated_secret("aioi"),
            allocated_storage=50,
            storage_type=rds.StorageType.GP3,
            deletion_protection=False,   # set True for production
            multi_az=False,              # Single-AZ for demo cost
        )
        # TODO: add custom resource to run CREATE EXTENSION IF NOT EXISTS vector
        # pgvector extension must be enabled after instance creation
```

### `infra/cdk/stacks/api_stack.py`

```python
from aws_cdk import Stack, aws_lambda as lambda_, aws_apigateway as apigw, Duration
from constructs import Construct

class ApiStack(Stack):
    def __init__(self, scope: Construct, id: str, vpc, **kwargs):
        super().__init__(scope, id, **kwargs)

        fn = lambda_.Function(self, "AIOIApiFunction",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="ai.api.handlers.main.handler",
            code=lambda_.Code.from_asset(".", bundling={
                "image": lambda_.Runtime.PYTHON_3_11.bundling_image,
                "command": ["bash","-c",
                    "pip install -r requirements.txt -t /asset-output && cp -r . /asset-output"],
            }),
            vpc=vpc,
            memory_size=512,
            timeout=Duration.seconds(30),
            architecture=lambda_.Architecture.ARM_64,  # Graviton — 20% cheaper
        )

        apigw.LambdaRestApi(self, "AIOIApiGateway", handler=fn)
```

### `infra/cdk/stacks/bedrock_stack.py`

```python
from aws_cdk import Stack
from constructs import Construct

class BedrockStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # TODO: attach Guardrails ARN to Bedrock Agent resource.
        # Configure: topic denial (non-insurance queries), PII redaction on output, grounding check.
        # Reference guardrail_arn from SSM Parameter Store or cdk.CfnParameter.
        # See CROSS_PHASE.md §9.2.
        pass
```

---

## 5. Deploy Commands

```bash
cd infra/cdk

# Synthesize — validate before deploy
cdk synth

# Deploy database first (VPC needed by API stack)
cdk deploy AIOI-Database

# Deploy API
cdk deploy AIOI-Api

# Deploy Bedrock configuration
cdk deploy AIOI-Bedrock

# Or deploy all at once
cdk deploy --all
```

```powershell
# Windows — same commands, run in PowerShell from infra/cdk/
cdk synth
cdk deploy --all
```

---

## 6. Intelligent Prompt Routing (Cost Optimization)

Add to `ai/agents/claims_agent/agent.py` before invoking Sonnet. Estimated saving: 30–50% of Oak Assist Bedrock cost.

```python
SIMPLE_PATTERNS = [
    r"what is my policy number",
    r"claim status",
    r"payment due",
    r"phone number",
    r"hours of operation",
]

def should_use_haiku(query: str) -> bool:
    """Route simple lookup queries to Haiku; complex FNOL to Sonnet."""
    import re
    q = query.lower()
    return any(re.search(p, q) for p in SIMPLE_PATTERNS)

def invoke_with_routing(session_id: str, user_message: str) -> str:
    model_id = (
        os.getenv("BEDROCK_MODEL_ID_HAIKU")
        if should_use_haiku(user_message)
        else os.getenv("BEDROCK_MODEL_ID_SONNET")
    )
    # ... invoke with selected model_id
```

See `iron_oak_insurance_strategy.md` Section 7.5 for the full ranked list of cost optimization levers.

---

## 7. Verification & Git Tag

### Verification

```bash
# After cdk deploy, get the API Gateway URL from CloudFormation outputs
API_URL=$(aws cloudformation describe-stacks \
  --stack-name AIOI-Api \
  --query "Stacks[0].Outputs[?OutputKey=='AIOIApiGatewayEndpoint'].OutputValue" \
  --output text)

# Test fraud scoring via Lambda
curl -X POST $API_URL/models/fraud/score \
  -H "Content-Type: application/json" \
  -d '{"claims": [...]}'

# Test RAG via Bedrock
curl -X POST $API_URL/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is my deductible on policy TX-00142?","mode":"bedrock"}'
```

```powershell
# Windows — get API Gateway URL
$API_URL = aws cloudformation describe-stacks `
  --stack-name AIOI-Api `
  --query "Stacks[0].Outputs[?OutputKey=='AIOIApiGatewayEndpoint'].OutputValue" `
  --output text
```

### Phase Gate Checklist

- [ ] `cdk synth` produces no errors
- [ ] Lambda cold start < 10s
- [ ] Bedrock model IDs validated (no legacy strings)
- [ ] Bedrock Guardrails ARN attached in CDK stack
- [ ] Guardrails topic denial tested with off-topic queries
- [ ] PII not present in CloudWatch Logs
- [ ] Intelligent prompt routing tested: simple query → Haiku, FNOL → Sonnet

### Git Tag

```bash
git add -A
git commit -m "Phase 5: Bedrock agents + CDK stacks — full cloud deployment"
git tag v1.0.0
```

---

*Previous: [PHASE_4_RAG.md](./PHASE_4_RAG.md) · Cross-phase decisions: [CROSS_PHASE.md](./CROSS_PHASE.md)*
