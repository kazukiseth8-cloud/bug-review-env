import uvicorn
from bug_review_env.server.app import app

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
