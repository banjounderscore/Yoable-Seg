# YoableSeg SAM3 with Grounding DINO Installation Script

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "YoableSeg SAM3 + Grounding DINO Installation" -ForegroundColor Cyan
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

    Write-Host "Installing Grounding DINO and SAM3 dependencies..." -ForegroundColor Gray
    & $pipPath install "transformers>=4.47.0" accelerate fastapi uvicorn pillow opencv-python numpy pydantic tqdm scipy 2>&1 | Out-Null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install packages" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host "Packages installed successfully" -ForegroundColor Green
} else {
    Write-Host "Packages already installed, skipping" -ForegroundColor Green
}

# HuggingFace login
Write-Host ""
Write-Host "[4/5] HuggingFace Authentication" -ForegroundColor Yellow
& hf auth login

# Download Models
Write-Host ""
Write-Host "[5/5] Downloading AI Models..." -ForegroundColor Yellow

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
    print('[1/2] Checking SAM3 (Segment Anything Model 3)...')
    print('  Using SAM3 instead of SAM2 as requested')
    if check_model_cached('facebook/sam3'):
        print('  SAM3 already downloaded, skipping')
        return True
    
    print('  Downloading SAM3 from HuggingFace (~3.5GB)...')
    print('  SAM3 is the latest segmentation model from Meta')
    print('  This may take several minutes...')
    try:
        from transformers import Sam3TrackerProcessor, Sam3TrackerModel
        
        print('  [1/2] Downloading SAM3 processor...')
        processor = Sam3TrackerProcessor.from_pretrained(
            'facebook/sam3', 
            trust_remote_code=True
        )
        
        print('  [2/2] Downloading SAM3 model (this is the large file)...')
        model = Sam3TrackerModel.from_pretrained(
            'facebook/sam3', 
            trust_remote_code=True, 
            ignore_mismatched_sizes=True
        )
        
        print('  SAM3 downloaded successfully!')
        return True
    except Exception as e:
        print(f'  WARNING: SAM3 download failed: {e}')
        return False

def download_grounding_dino():
    print('')
    print('[2/2] Checking GroundingDINO Base (Text-to-Detection)...')
    print('  GroundingDINO enables detection with text prompts')
    if check_model_cached('IDEA-Research/grounding-dino-base'):
        print('  GroundingDINO Base already downloaded, skipping')
        return True
    
    print('  Downloading GroundingDINO Base from HuggingFace (~933MB)...')
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        print('  [1/2] Downloading GroundingDINO Base processor...')
        processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
        
        print('  [2/2] Downloading GroundingDINO Base model...')
        model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
        
        print('  GroundingDINO Base downloaded successfully!')
        return True
    except Exception as e:
        print(f'  WARNING: GroundingDINO Base download failed: {e}')
        return False

print('')
print('Downloading AI models for YoableSeg...')
print('Pipeline: Grounding DINO (detect) -> SAM3 (segment)')
print('This may take several minutes on first run.')
print('')

sam3_ok = download_sam3()
dino_ok = download_grounding_dino()

if not sam3_ok or not dino_ok:
    print('')
    print('WARNING: Some models failed to download.')
    sys.exit(1)

print('')
print('All models downloaded successfully!')
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
    Write-Host "WARNING: Some models failed to download. They will be downloaded on first use." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
