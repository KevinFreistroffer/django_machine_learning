#!/usr/bin/env bash
# Exit on error
set -o errexit

# Debug: Print current directory
pwd
ls
ls -la

cd django_machine_learning/app
ls

pip freeze > requirements.txt
# Modify this line as needed for your package manager (pip, poetry, etc.)
pip install -r requirements.txt

# Convert static asset files
python manage.py collectstatic --no-input

# Apply any outstanding database migrations
python manage.py migrate