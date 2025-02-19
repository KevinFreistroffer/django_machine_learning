#!/usr/bin/env bash
# Exit on error
set -o errexit
pip freeze > requirements.txt
# Modify this line as needed for your package manager (pip, poetry, etc.)
pip install -r requirements.txt

# Convert static asset files
python django_machine_learning/app/manage.py collectstatic --no-input

# Apply any outstanding database migrations
python django_machine_learning/app/manage.py migrate