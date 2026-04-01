"""
OpenEnv spec-compliant entry point.
Delegates to my_env.server.app for the core implementation.
"""
from my_env.server.app import app as support_app, main as support_main

# Required by OpenEnv validator
app = support_app

def main():
    """Start the FastAPI server."""
    support_main()

if __name__ == "__main__":
    main()
