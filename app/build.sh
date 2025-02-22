#!/bin/bash
# Exit on error
set -o errexit

echo "Building..."

# Debug: Print current directory
pwd
ls
ls -la

echo "Changing directory..."
cd app
ls

echo "Creating virtual environment..."
python -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

# echo "Freezing requirements..."
# pip freeze > requirements.txt

echo "Installing requirements..."
# Modify this line as needed for your package manager (pip, poetry, etc.)
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating staticfiles directory..."
mkdir -p staticfiles

echo "Collecting static assets..."
python manage.py collectstatic --no-input --clear

echo "Applying database migrations..."
python manage.py migrate

echo "Build completed."