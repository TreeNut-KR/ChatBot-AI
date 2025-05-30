from pathlib import Path

if __name__ == "__main__":
    env_file_path = Path(__file__).resolve().parents[3] / ".env"
    print(f"Current directory: {env_file_path}")