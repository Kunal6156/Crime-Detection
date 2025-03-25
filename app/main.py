import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.api import run_api_server  # Adjusted import

def main():
    try:
        run_api_server()
    except Exception as e:
        print(f"Error starting API server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()