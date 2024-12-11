# Base Image
FROM python:3.10-slim

# Set Working Directory
WORKDIR /app

# Copy Dependencies and Code
COPY requirements.txt .
COPY llama3B_SAE_train.py .

# Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Configure Hugging Face and Wandb using Secrets
# Hugging Face Token
# Use Docker secrets to authenticate with Hugging Face CLI
RUN --mount=type=secret,id=hf_token \
    huggingface-cli login --token "$(cat /run/secrets/hf_token)"

# Wandb Configuration (if needed)
RUN --mount=type=secret,id=wandb_token \
    wandb login --relogin "$(cat /run/secrets/wandb_token)"

# Download llama-3.2-3B-Instruct model
RUN huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir /app/models/llama-3.2-3B-Instruct --token "$(cat /run/secrets/hf_token)"

# Step 8: Default Command
CMD ["python", "llama3B_SAE_train.py"]