FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY engines/ ./engines/
COPY loader/ ./loader/
COPY pipeline/ ./pipeline/
COPY prediction/ ./prediction/
COPY training/ ./training/
COPY models/ ./models/
COPY data/ ./data/
COPY utils/ ./utils/
COPY evaluation/ ./evaluation/
COPY validation/ ./validation/

CMD ["prediction.handler.lambda_handler"]



# FROM public.ecr.aws/lambda/python:3.11

# WORKDIR /var/task

# # Use pip mirror and timeout
# COPY requirements.txt .
# RUN pip install --no-cache-dir --prefer-binary -r requirements.txt


# COPY . .

# CMD ["prediction.handler.lambda_handler"]



# # FROM public.ecr.aws/lambda/python:3.11

# # WORKDIR /var/task

# # # Install dependencies - scipy 1.10.1 has pre-built wheels
# # COPY requirements.txt .
# # RUN pip install --no-cache-dir -r requirements.txt

# # # Copy application
# # COPY . .

# # CMD ["prediction.handler.lambda_handler"]