# 🐋 ETH Whale Alpha — Price Direction Generator

A machine learning system that generates daily trading signals for Ethereum (ETH) using whale activity, market data, and regime-aware models. The project is fully deployable on AWS Lambda using Docker and exposes predictions through a public API.

---

## 📌 Project Overview

Large cryptocurrency holders ("whales") often move markets before price reacts. This project answers the question:

**Can we use whale activity and market structure to predict ETH price direction in a reliable, production-ready way?**

This system combines:
- **On-chain whale behavior**
- **Market indicators**
- **Machine learning**
- **Regime detection**
- **Cloud deployment**

to generate practical trading signals for ETH.

---

## 🎯 What This Project Does

- ✅ Processes historical ETH whale and market data
- ✅ Engineers predictive features
- ✅ Detects market regimes
- ✅ Trains separate LONG and SHORT models
- ✅ Combines ML with rule-based logic using a Core Engine
- ✅ Produces daily trade signals
- ✅ Runs locally, in Docker, or in AWS Lambda
- ✅ Exposes predictions via a public HTTP endpoint

---

## 🧠 How It Works

**High-level flow:**

```
Data → Feature Engineering → Regime Detection  
     → LONG/SHORT Models → CoreTrendEngine  
     → Prediction API (Lambda)
```

The system adapts its behavior based on market regime and only issues trades when confidence is sufficient.

---

## 📂 Project Structure

```
Intent/
│
├── loader/          # Data loading & feature engineering
├── training/        # Model training logic
├── engines/         # CoreTrendEngine (signal logic)
├── models/          # Trained ML models
├── pipeline/        # Data preparation pipeline
├── prediction/      # AWS Lambda handler
├── shadow_trading/  # Forward testing & simulation
├── validation/      # Model evaluation
├── main.py          # Local entry point
└── Dockerfile       # Container definition
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <https://github.com/FirstBML/Whale-Movement-Based-Price-Direction-Generator-V2.git>
cd WhalesIntent/Intent
```

### 2. Run Locally (Python)

**Activate environment:**

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**Install dependencies:**

```bash
uv sync
```

**Run the engine:**

```bash
python main.py
```

---

## 🐳 Run with Docker (Lambda Emulator)

### Build the Image

```bash
docker build -t whale-alpha .
```

### Run Locally

```bash
docker run -p 9000:8080 whale-alpha
```

### Test the API (PowerShell)

```powershell
$response = Invoke-RestMethod -Uri "http://localhost:9000/2015-03-31/functions/function/invocations" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{}'

$response | ConvertTo-Json -Depth 10
```

---

## ☁️ Cloud Deployment (AWS Lambda)

The system is deployed using Docker-based AWS Lambda.

### Deployment Flow

1. Build Docker image
2. Push to Amazon ECR
3. Create Lambda from container
4. Attach IAM role
5. Expose via API Gateway or Function URL

### Lambda Handler

```
prediction.handler.lambda_handler
```

---

## 🌐 Public API (Example)

```
https://o7c8rqely2.execute-api.us-east-1.amazonaws.com/prod
```

You can open this in a browser or test with:

```bash
curl https://o7c8rqely2.execute-api.us-east-1.amazonaws.com/prod/predict
```

---

## 🧪 Sample Output

```json
{
  "date": "2026-01-06",
  "regime": "R2",
  "eth_price": 3233.17,
  "btc_price": 93288.34,
  "signal": {
    "action": "NO_TRADE",
    "direction": null,
    "confidence": 0.0,
    "position_size": 0.0,
    "model_probability": 0.0,
    "reasons": ["neutral_regime"],
    "engine": "core_trend"
  }
}
```

---

## 📦 Docker Setup

### Dockerfile

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["prediction.handler.lambda_handler"]
```

**Build:**

```bash
docker build --platform linux/amd64 -t whale-alpha .
```

---

## ✅ What Has Been Achieved

- ✅ Regime-aware ML trading engine
- ✅ LONG and SHORT model pipelines
- ✅ Calibrated probabilities
- ✅ Shadow trading (forward testing)
- ✅ Clean production-ready architecture
- ✅ Dockerized deployment
- ✅ AWS Lambda live deployment
- ✅ Public prediction endpoint
- ✅ Hybrid local + S3 data loading
- ✅ Reviewer-ready codebase

---

## 🔧 What Still Needs to Be Done

- ⏳ Live market data ingestion
- ⏳ Automated retraining pipeline
- ⏳ CI/CD for model updates
- ⏳ Monitoring dashboard
- ⏳ Feature drift detection
- ⏳ Position sizing optimization
- ⏳ Multi-asset support
- ⏳ Optional web frontend

---

## 💡 How This Can Be Used

This project can serve as:

- A crypto trading signal generator
- A quantitative research framework
- A cloud-deployed ML system
- A portfolio project for ML + Web3 + MLOps
- A base for institutional-grade trading systems

---

## 👤 Author

**Bashiru ML**  
Data Analyst & On-chain Alpha Researcher

**Focus areas:**
- On-chain analytics
- Quant trading systems
- DeFi market structure
- Machine Learning in finance

---

## 📜 License

This project is for academic and research purposes.  
Commercial use requires permission.