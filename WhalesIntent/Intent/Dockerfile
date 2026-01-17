# Build stage
FROM public.ecr.aws/lambda/python:3.11 as builder

WORKDIR ${LAMBDA_TASK_ROOT}

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --quiet -r requirements.txt

# Runtime stage
FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

# Copy only installed packages from builder
COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}

# Copy application code
COPY . .

CMD ["prediction.handler.lambda_handler"]