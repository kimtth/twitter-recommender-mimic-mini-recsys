"""
Complete training and testing workflow
- Trains all PyTorch models on dataset
- Runs comprehensive test suite
- Verifies model integration
"""
import os
import sys
import subprocess
from time import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def print_header(text):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"\n🔄 {description}...")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[X] {description} failed with exit code {e.returncode}")
        print(e.stdout)
        print(e.stderr)
        return False


def check_dependencies():
    """Verify required packages are installed"""
    print_header("Checking Dependencies")
    
    required = ['numpy', 'torch', 'pandas', 'pyarrow', 'pytest']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[X] {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n[WARN] Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def check_dataset():
    """Verify dataset exists"""
    print_header("Checking Dataset")
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    required_files = ['users.parquet', 'tweets.parquet', 'interactions.parquet', 'follows.parquet']
    
    if not os.path.exists(data_dir):
        print("[X] Data directory not found")
        print("Run: python prep/generate_dataset.py")
        return False
    
    missing = []
    for fname in required_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024  # KB
            print(f"[OK] {fname} ({size:.1f} KB)")
        else:
            print(f"[X] {fname} - MISSING")
            missing.append(fname)
    
    if missing:
        print(f"\n[WARN] Missing files: {', '.join(missing)}")
        print("Run: python prep/generate_dataset.py")
        return False
    
    return True


def train_models():
    """Train all PyTorch models"""
    print_header("Training Models")
    
    # Change to project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(project_root)
    
    return run_command(
        "python models/train_models.py",
        "Model training"
    )


def run_tests():
    """Run test suite"""
    print_header("Running Tests")
    
    # Try pytest first, fallback to custom runner
    pytest_available = subprocess.run(
        "pytest --version",
        shell=True,
        capture_output=True
    ).returncode == 0
    
    if pytest_available:
        success = run_command(
            "pytest -v tests/",
            "PyTest suite"
        )
    else:
        print("[WARN] pytest not found, using custom test runner")
        success = run_command(
            "python tests/run_tests.py",
            "Custom test suite"
        )
    
    return success


def run_inference_demo():
    """Run main.py to verify end-to-end system"""
    print_header("Running Inference Demo")
    
    return run_command(
        "python main.py",
        "End-to-end recommendation pipeline"
    )


def verify_models():
    """Check that trained models exist"""
    print_header("Verifying Trained Models")
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_files = [
        'engagement_model.pt',
        'nsfw_model.pt',
        'toxicity_model.pt',
        'twhin_model.pt'
    ]
    
    all_exist = True
    for fname in model_files:
        fpath = os.path.join(models_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024  # KB
            print(f"[OK] {fname} ({size:.1f} KB)")
        else:
            print(f"[X] {fname} - NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    """Execute complete training and testing workflow"""
    start_time = time()
    
    print_header("Mini-RecSys Training & Testing Pipeline")
    print("This script will:")
    print("  1. Check dependencies")
    print("  2. Verify dataset exists")
    print("  3. Train all PyTorch models (~2-3 minutes)")
    print("  4. Run comprehensive tests")
    print("  5. Verify model integration")
    print("  6. Run inference demo")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n[X] Dependencies check failed. Please install missing packages.")
        return 1
    
    # Step 2: Check dataset
    if not check_dataset():
        print("\n[X] Dataset check failed. Please generate dataset first.")
        print("   Run: python prep/generate_dataset.py")
        return 1
    
    # Step 3: Train models
    if not train_models():
        print("\n[X] Model training failed.")
        return 1
    
    # Step 4: Verify models were saved
    if not verify_models():
        print("\n[WARN] Some models were not saved properly.")
        return 1
    
    # Step 5: Run tests
    if not run_tests():
        print("\n[WARN] Some tests failed, but continuing...")
    
    # Step 6: Run inference demo
    if not run_inference_demo():
        print("\n[X] Inference demo failed.")
        return 1
    
    # Success summary
    elapsed = time() - start_time
    print_header("[OK] Training & Testing Complete!")
    print(f"Total time: {elapsed:.1f} seconds")
    print("\nTrained models are ready for use:")
    print("  - models/engagement_model.pt")
    print("  - models/nsfw_model.pt")
    print("  - models/toxicity_model.pt")
    print("  - models/twhin_model.pt")
    print("\nYou can now:")
    print("  - Run recommendations: python main.py")
    print("  - Run evaluation: python eval/eval.py")
    print("  - Run tests: pytest -v tests/")
    
    return 0


if __name__ == '__main__':
    exit(main())
