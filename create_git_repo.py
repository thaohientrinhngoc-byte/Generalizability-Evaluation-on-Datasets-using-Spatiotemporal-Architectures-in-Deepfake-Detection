"""
Automated Local Git Repository Setup Script.
"""

import os
import shutil
import subprocess

SOURCE_DIR = r"C:\Users\LENOVO\.gemini\antigravity-ide\scratch\deepfake-spatiotemporal-generalizability"
TARGET_DIR = r"C:\Users\LENOVO\Downloads\Paper deepfake\deepfake-spatiotemporal-generalizability"


def copy_and_init_git(repo_dir: str):
    print(f"\n--- Initializing Git Repository in: {repo_dir} ---")
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir, exist_ok=True)

    # Initialize Git
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Deepfake Research Team"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "thaohientrinhngoc@gmail.com"], cwd=repo_dir, check=True)

    # Add & Commit
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit: Spatiotemporal Deepfake Detection Generalizability Framework"], cwd=repo_dir, check=True)
    print(f"Git repository successfully initialized with initial commit at {repo_dir}")


def main():
    print(f"Copying repository from scratch to Downloads folder...")
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR, ignore_errors=True)

    shutil.copytree(SOURCE_DIR, TARGET_DIR, ignore=shutil.ignore_patterns('.git', '__pycache__'), dirs_exist_ok=True)
    print(f"Copied files to: {TARGET_DIR}")

    # Init Git in both directories
    copy_and_init_git(SOURCE_DIR)
    copy_and_init_git(TARGET_DIR)


if __name__ == "__main__":
    main()
