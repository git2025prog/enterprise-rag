# Enterprise RAG Starter (FAISS + OpenAI + Streamlit)

A tiny end-to-end Retrieval-Augmented Generation (RAG) starter you can run in minutes. It:

- embeds your docs with **OpenAI** embeddings  
- stores vectors in **FAISS**  
- retrieves the most relevant chunks  
- calls an LLM to answer **strictly from your docs**  
- ships with a minimal **Streamlit** UI

---

## Table of contents

- [Quick start (local)](#quick-start-local)  
- [How it works](#how-it-works)  
- [CLI demo (optional)](#cli-demo-optional)  
- [Run with Docker (local)](#run-with-docker-local)  
- [Deploy on AWS (EC2 + Docker)](#deploy-on-aws-ec2--docker)  
- [Project layout](#project-layout)  
- [Notes & troubleshooting](#notes--troubleshooting)  
- [License](#license)

---

## Quick start (local)

**Prereqs:** Python 3.10+ and an OpenAI API key.

1) Create a venv & install deps
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

2) Set your API key (pick one)
```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```
> Prefer env files? Create `.env` with `OPENAI_API_KEY=sk-...` (already in `.gitignore`).

3) Put a few `.txt` / `.md` / `.pdf` files in the **data/** folder (a sample is included).

4) Run the UI
```bash
streamlit run streamlit_app.py
```

In the app:
- click **Ingest ./data** (builds a FAISS index under `store/`)
- ask a question
- see the answer plus **Top passages** used for grounding

---

## How it works

1. **Embeddings** — `rag_core.embed_texts()` turns text into vectors (OpenAI embeddings).  
2. **Indexing** — `rag_core.ingest_directory()` chunks + stores vectors in **FAISS**.  
3. **Retrieval** — `rag_core.retrieve()` searches FAISS and maps IDs back to source snippets via `meta.jsonl`.  
4. **LLM answer** — `rag_core.answer()` builds a grounded prompt with retrieved context and calls the chat model.

> Vectors are normalized and stored in `IndexFlatIP` (inner product), which approximates cosine similarity.

---

## CLI demo (optional)

Run a tiny non-UI demo:

```bash
python simple_cli_demo.py
```

---

## Run with Docker (local)

Build an image and run the container mounting your local `data/`:

```bash
# Build image
docker build -t rag-app:latest .

# Run (macOS/Linux)
docker run --rm -p 8501:8501   -e OPENAI_API_KEY=$OPENAI_API_KEY   -v "$(pwd)/data:/app/data"   rag-app:latest

# Run (Windows PowerShell)
docker run --rm -p 8501:8501 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -v "${PWD}\data:/app/data" `
  rag-app:latest
```

Open http://localhost:8501 and click **Ingest ./data**.

---

## Deploy on AWS (EC2 + Docker)

> **What this stack does:** launches an EC2 instance with Docker, pulls your **ECR** image, injects your OpenAI key from **SSM Parameter Store**, and exposes Streamlit on port **8501**.

### 1) Push your image to ECR
```bash
# set variables
REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# repo (idempotent)
aws ecr create-repository --repository-name rag-app --region $REGION || true

# login, tag, push
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR
docker tag rag-app:latest $ECR/rag-app:latest
docker push $ECR/rag-app:latest
```

### 2) Store your OpenAI key in SSM Parameter Store
```bash
aws ssm put-parameter   --name "/rag/openai"   --value "sk-***"   --type SecureString   --overwrite   --region us-east-1
```

### 3) Launch the CloudFormation stack
Use `rag-ec2.yaml` (CloudFormation → **Create stack** → **Upload a template**), then set parameters:

- **AmiId**: `/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64` (SSM param)  
- **EcrRepoUri**: `ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-app:latest`  
- **InstanceType**: `t3.small` (fine for demos)  
- **KeyName**: *(EC2 key pair — optional if using Session Manager)*  
- **SsmParamName**: `/rag/openai`  
- **VpcId** / **SubnetId**: a public subnet (or any with egress + SSM access)

When CREATE_COMPLETE, open:  
`http://<EC2_PUBLIC_DNS>:8501`

**Tip:** If your source files are in S3, copy them to the instance and mount:
```bash
# on the EC2 instance (e.g., via Session Manager)
sudo mkdir -p /srv/rag-data
aws s3 sync s3://YOUR_BUCKET /srv/rag-data
# restart container with:  -v /srv/rag-data:/app/data
```

---

## Project layout

```
.
├─ rag_core.py           # ingest, retrieve, answer
├─ streamlit_app.py      # Streamlit UI
├─ simple_cli_demo.py    # tiny CLI demo
├─ rag-ec2.yaml          # CloudFormation for EC2+Docker deploy
├─ Dockerfile
├─ requirements.txt
├─ data/                 # put PDFs/TXTs/MDs here
└─ store/                # FAISS index + metadata (created on ingest; .gitignored)
```

---

## Notes & troubleshooting

- If you see **“Index not found”**, click **Ingest ./data** first to build the FAISS index.  
- The app will answer **“I don’t know.”** for questions outside your corpus (or with low retrieval confidence).  
- For production: use token-aware chunking, enrich metadata filters, consider a reranker, and move to a persistent vector DB.

---

## License

MIT (or your preferred license). Add a `LICENSE` file if publishing publicly.

---

### (Optional) Screenshots

- Travel meals policy — grounded answer  
- Expense policy — grounded answer  
- Out-of-corpus question — **“I don’t know.”**  
- CloudFormation stack **CREATE_COMPLETE**  
- S3 bucket with example docs  

> **Security:** blur/crop Account IDs, ARNs, bucket names, public IP/DNS, and repo URIs before sharing screenshots.
