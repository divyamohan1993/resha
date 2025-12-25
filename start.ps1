$ErrorActionPreference = "Stop"

function Write-LogInfo { Param($Msg) Write-Host "[INFO] $(Get-Date -Format 'HH:mm:ss') - $Msg" -ForegroundColor Green }
function Write-LogFailure { Param($Msg) Write-Host "[ERROR] $(Get-Date -Format 'HH:mm:ss') - $Msg" -ForegroundColor Red; exit 1 }

# 1. CHECK WSL
$wslStatus = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslStatus) {
    Write-LogFailure "WSL is not installed. Please install WSL 2 to use this agent without polluting your Windows environment."
}

# 2. BRIDGE TO WSL (Clean & Isolated)
Write-LogInfo "Bridging to WSL for isolated execution..."
Write-LogInfo "The agent will handle everything inside the Linux subsystem."

# Convert current path to WSL path
$currentPath = Get-Location
$wslPath = "/mnt/" + $currentPath.Drive.Name.ToLower().TrimEnd(':') + $currentPath.Path.Substring(2).Replace('\', '/')

# Execute the master script inside WSL
wsl bash -c "cd '$wslPath' && bash start.sh --force-wsl"

if ($LASTEXITCODE -ne 0) {
    Write-LogFailure "Deployment failed inside WSL."
}
