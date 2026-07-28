#Requires -Version 5.1

<#
.SYNOPSIS
    Build single-file executable of deploy.py via PyInstaller.
    Interactive prompts for runtime (onnxruntime / onnxruntime-gpu) and version.
#>

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $PSCommandPath
Set-Location $ScriptDir

$VenvPython = ".venv\Scripts\python.exe"
$VenvPip    = ".venv\Scripts\pip.exe"
$VenvPyInst = ".venv\Scripts\pyinstaller.exe"

# ---- check venv ----
if (-not (Test-Path $VenvPython)) {
    Write-Error "venv not found at $VenvPython — create it with 'uv venv' first"
    exit 1
}

# ---- interactive runtime selection ----
Write-Host "Select ONNX runtime:"
Write-Host "  1) onnxruntime        (CPU only)"
Write-Host "  2) onnxruntime-gpu     (GPU/CUDA) [default]"
$choice = Read-Host "Choice [1/2] (default=2)"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = '2' }

$RuntimePkg = if ($choice -eq '1') { 'onnxruntime' } else { 'onnxruntime-gpu' }
$CudaTag = if ($choice -eq '1') { '_cpu' } else { '' }

$ver = Read-Host "Version (default=1.18.0)"
if ([string]::IsNullOrWhiteSpace($ver)) { $ver = '1.18.0' }

Write-Host "Installing ${RuntimePkg}==${ver} + numpy + Pillow + gradio ..."
& $VenvPip install --quiet "${RuntimePkg}==${ver}", 'numpy', 'Pillow', 'gradio', 'pyinstaller'
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host ""

# ---- extract version from pyproject.toml ----
$toml = Get-Content 'pyproject.toml' -Raw
$match = [regex]::Match($toml, "^version\s*=\s*`"(?<ver>[^`"]+)`"", [System.Text.RegularExpressions.RegexOptions]::Multiline)
if (-not $match.Success) {
    Write-Error "Could not extract version from pyproject.toml"
    exit 1
}
$Version = $match.Groups['ver'].Value

# ---- detect OS & arch ----
$OS = 'Windows'
$Arch = switch ($env:PROCESSOR_ARCHITECTURE) {
    'AMD64'  { 'x86_64' }
    'ARM64'  { 'arm64'  }
    'x86'    { 'x86'    }
    default  { $env:PROCESSOR_ARCHITECTURE.ToLower() }
}

# ---- detect CUDA version via torch (only if GPU runtime) ----
if ($choice -eq '2') {
    $CudaTag = & $VenvPython -c "import torch; v=torch.version.cuda; print(f'_cuda{v}' if v else '_cpu')" 2>$null
    if (-not $CudaTag) { $CudaTag = '_cpu' }
}

$Name = "DanbooruTagQuery_${Version}_${OS}_${Arch}${CudaTag}"

Write-Host "Building: ${Name}"
Write-Host "  Version: ${Version}"
Write-Host "  OS:      ${OS}"
Write-Host "  Arch:    ${Arch}"
if ($CudaTag -eq '_cpu') {
    Write-Host "  CUDA:    none (cpu)"
} else {
    Write-Host "  CUDA:    $($CudaTag.Substring(1))"
}

# ---- build PyInstaller arguments ----
$pyiArgs = @(
    '--onefile'
    '--name', $Name
    '--distpath', 'dist'
    '--workpath', 'build\pyinstaller'
    '--specpath', 'build\pyinstaller'
)

$pyiArgs += 'deploy.py'

& $VenvPyInst @pyiArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
