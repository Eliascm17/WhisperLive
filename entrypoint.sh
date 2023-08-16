#!/bin/sh
set -e

if [ "$ENV" = 'DEV' ]; then
    echo "Running Development Server"
    exec uvicorn main:app --host 0.0.0.0 --port 9000 --reload
else
    echo "Running Production Server"
    exec uvicorn main:app --host 0.0.0.0 --port 9000
fi
