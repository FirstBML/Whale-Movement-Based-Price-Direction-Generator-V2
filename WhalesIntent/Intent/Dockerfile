FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

# Install dependencies - scipy 1.10.1 has pre-built wheels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

CMD ["prediction.handler.lambda_handler"]