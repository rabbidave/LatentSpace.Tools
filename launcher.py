#!/usr/bin/env python3
"""
Model A/B Testing Tool Launcher using uv

This script will:
1. Check if uv is installed, install it if needed
2. Create a virtual environment with uv
3. Install dependencies in the venv
4. Launch the main testing application
"""

import subprocess
import sys
import os
import platform
import venv
import tempfile
import shutil

# Required packages for the A/B testing tool
REQUIREMENTS = [
    "gradio>=4.0.0",
    "pandas>=1.3.0",
    "requests>=2.25.0",
    "tenacity>=8.0.0",
    "httpx>=0.24.0"
]

# Where to create the venv
VENV_DIR = os.path.join(os.path.expanduser("~"), ".model-ab-tester-venv")

def check_uv_installed():
    """Check if uv is installed and install it if needed."""
    try:
        subprocess.check_call(["uv", "--version"], stdout=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("uv not found. Installing uv...")
        try:
            # Install uv using the recommended method
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
            return True
        except subprocess.CalledProcessError:
            print("Failed to install uv with pip. Trying with curl...")
            try:
                # For Unix-based systems, try the curl installer
                if platform.system() != "Windows":
                    subprocess.check_call(
                        ["curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"],
                        shell=True
                    )
                    return True
                else:
                    print("Please install uv manually from https://github.com/astral-sh/uv")
                    return False
            except subprocess.CalledProcessError:
                print("Failed to install uv. Please install it manually.")
                return False

def get_venv_python():
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

def setup_environment():
    """Create and set up a virtual environment using uv."""
    # Check if venv exists
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment at {VENV_DIR}...")
        venv.create(VENV_DIR, with_pip=True)
    
    # Install dependencies with uv
    print(f"Installing dependencies with uv...")
    requirements_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
    try:
        # Write requirements to a temporary file
        for req in REQUIREMENTS:
            requirements_file.write(f"{req}\n")
        requirements_file.close()
        
        # Use uv to install packages into the venv
        subprocess.check_call([
            "uv", "pip", "install", 
            "--requirement", requirements_file.name,
            "--python", get_venv_python()
        ])
    finally:
        # Clean up temp file
        os.unlink(requirements_file.name)
    
    return True

def run_main_script():
    """Run the main testing script using the venv Python."""
    print("\nLaunching Model A/B Testing Tool...")
    venv_python = get_venv_python()
    
    # Copy the main script to the venv directory to avoid escape sequence issues
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-ab-testing-gradio.py")
    venv_script_path = os.path.join(VENV_DIR, "model-ab-testing-gradio.py")
    
    # Fix escape sequences when copying the file
    with open(script_path, 'r', encoding='utf-8') as source:
        content = source.read()
    
    # Fix the escape sequences in the script if needed
    # This is a simplistic approach - a more robust parser might be needed
    # for complex escape sequence fixes
    content = content.replace(r'\s', r'\\s')
    content = content.replace(r'\n', r'\\n')
    content = content.replace(r'\t', r'\\t')
    content = content.replace(r'\x', r'\\x')
    
    with open(venv_script_path, 'w', encoding='utf-8') as dest:
        dest.write(content)
    
    try:
        # Run the script with the venv Python
        result = subprocess.run([venv_python, venv_script_path] + sys.argv[1:])
        return result.returncode
    except Exception as e:
        print(f"Error launching the application: {str(e)}")
        return 1

if __name__ == "__main__":
    if check_uv_installed() and setup_environment():
        sys.exit(run_main_script())
    else:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)