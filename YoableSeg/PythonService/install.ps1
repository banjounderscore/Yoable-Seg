# YoableSeg SAM3 Installation Script

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "YoableSeg SAM3 Installation" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python not found. Please install Python 3.10-3.13 first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$pythonVersion = (python --version 2>&1) -replace "Python ", ""
Write-Host "Found Python $pythonVersion" -ForegroundColor Green

# Setup paths
$serviceDir = "$env:LOCALAPPDATA\YoableSeg\PythonService"
$venvPath = Join-Path $serviceDir "venv"
$pipPath = Join-Path $venvPath "Scripts\pip.exe"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

# Check if venv exists and has torch installed
Write-Host ""
Write-Host "[2/5] Checking virtual environment..." -ForegroundColor Yellow

$needsInstall = $true
if (Test-Path $pythonExe) {
    $torchCheck = & $pythonExe -c "import torch; print('ok')" 2>$null
    if ($torchCheck -eq "ok") {
        Write-Host "Virtual environment exists with packages installed" -ForegroundColor Green
        $needsInstall = $false
    } else {
        Write-Host "Virtual environment exists but packages missing, reinstalling..." -ForegroundColor Gray
        Remove-Item -Path $venvPath -Recurse -Force
    }
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
}

if ($needsInstall) {
    New-Item -ItemType Directory -Path $serviceDir -Force | Out-Null
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Virtual environment created" -ForegroundColor Green
}

# Install packages
Write-Host ""
Write-Host "[3/5] Installing Python packages..." -ForegroundColor Yellow

if ($needsInstall) {
    Write-Host "This may take 5-10 minutes" -ForegroundColor Gray

    Write-Host "Upgrading pip..." -ForegroundColor Gray
    & $pythonExe -m pip install --upgrade pip wheel setuptools 2>&1 | Out-Null

    Write-Host "Installing PyTorch (CUDA 12.1)..." -ForegroundColor Gray
    & $pipPath install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | Out-Null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install PyTorch" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host "Installing SAM3 dependencies..." -ForegroundColor Gray
    & $pipPath install "transformers>=4.47.0" accelerate fastapi uvicorn pillow opencv-python numpy pydantic tqdm scipy shapely 2>&1 | Out-Null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install packages" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host "Packages installed successfully" -ForegroundColor Green
} else {
    Write-Host "Packages already installed, skipping" -ForegroundColor Green
}

# HuggingFace login (SAM3 is a gated model)
Write-Host ""
Write-Host "[4/5] HuggingFace Authentication" -ForegroundColor Yellow
Write-Host "SAM3 requires HuggingFace authentication (gated model)" -ForegroundColor Gray
Write-Host "If you haven't already, accept the license at: https://huggingface.co/facebook/sam3" -ForegroundColor Gray

$hfPath = Join-Path $venvPath "Scripts\huggingface-cli.exe"
if (Test-Path $hfPath) {
    & $hfPath login
} else {
    Write-Host "Installing huggingface_hub..." -ForegroundColor Gray
    & $pipPath install huggingface_hub 2>&1 | Out-Null
    & $pythonExe -m huggingface_hub.commands.huggingface_cli login
}

# Download Models
Write-Host ""
Write-Host "[5/5] Downloading SAM3 Model..." -ForegroundColor Yellow

$scriptPath = Join-Path $serviceDir "download_models.py"
@"
import sys
import os
from tqdm import tqdm

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def check_model_cached(model_id):
    try:
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        model_dir = f'models--{model_id.replace("/", "--")}'
        full_path = os.path.join(cache_dir, model_dir)
        if os.path.exists(full_path):
            refs_path = os.path.join(full_path, 'refs', 'main')
            return os.path.exists(refs_path)
        return False
    except:
        return False

def download_sam3():
    print('Checking SAM3 (Segment Anything Model 3)...')
    print('  SAM3 supports text-prompt concept segmentation')
    if check_model_cached('facebook/sam3'):
        print('  SAM3 already downloaded, skipping')
        return True
    
    print('  Downloading SAM3 from HuggingFace...')
    print('  SAM3 is the latest segmentation model from Meta')
    print('  This may take several minutes...')
    try:
        print('  [1/2] Downloading SAM3 processor...')
        from transformers import Sam3Processor
        processor = Sam3Processor.from_pretrained(
            'facebook/sam3', 
            trust_remote_code=True
        )
        
        print('  [2/2] Downloading SAM3 model (this is the large file)...')
        from transformers import Sam3Model
        model = Sam3Model.from_pretrained(
            'facebook/sam3', 
            trust_remote_code=True, 
            ignore_mismatched_sizes=True
        )
        
        print('  SAM3 downloaded successfully!')
        return True
    except Exception as e:
        print(f'  WARNING: SAM3 download failed: {e}')
        import traceback
        traceback.print_exc()
        return False

print('')
print('Downloading SAM3 for YoableSeg...')
print('SAM3 enables text-prompt auto-labeling (e.g., "person", "car")')
print('This may take several minutes on first run.')
print('')

sam3_ok = download_sam3()

if not sam3_ok:
    print('')
    print('WARNING: SAM3 failed to download.')
    sys.exit(1)

print('')
print('SAM3 downloaded successfully!')
"@ | Set-Content -Path $scriptPath -Encoding UTF8

& $pythonExe $scriptPath
$downloadResult = $LASTEXITCODE

Remove-Item $scriptPath -ErrorAction SilentlyContinue

if ($downloadResult -eq 0) {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host "Installation Complete!" -ForegroundColor Green
    Write-Host "==================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "WARNING: SAM3 failed to download. It will be downloaded on first use." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
