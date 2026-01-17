FROM public.ecr.aws/lambda/python:3.13

# Copy dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy project
COPY . .

# Lambda entrypoint
CMD ["prediction.handler.lambda_handler"]
