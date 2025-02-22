# Start with a Python image that includes just what we need
FROM python:3.10-slim

# Set up a working directory in the container
WORKDIR /app

# Copy and install Python requirements first (for better caching)
COPY app/requirements.txt app/requirements.txt
COPY pytorch/requirements.txt pytorch/requirements.txt

RUN pip install -r app/requirements.txt && \
    pip install -r pytorch/requirements.txt

# Copy the rest of your app code
COPY . .

# Tell Docker that your app uses port 8000
EXPOSE 8000

# The command to start your app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"] 