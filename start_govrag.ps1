[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$EnvPath = Join-Path $ProjectRoot ".env"

$EmbeddingAlias = "qwen3-embedding-0.6b"
$ChatAlias = "qwen2.5-1.5b"

function Get-DotEnvValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $escapedName = [Regex]::Escape($Name)
    $line = Get-Content -Path $EnvPath |
        Where-Object { $_ -match "^$escapedName=" } |
        Select-Object -First 1

    if (-not $line) {
        return $null
    }

    return ($line -split "=", 2)[1].Trim()
}

function Set-DotEnvValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $content = [IO.File]::ReadAllText($EnvPath)
    $escapedName = [Regex]::Escape($Name)
    $pattern = "(?m)^$escapedName=.*$"
    $replacement = "$Name=$Value"

    if ([Regex]::IsMatch($content, $pattern)) {
        $content = [Regex]::Replace($content, $pattern, $replacement)
    }
    else {
        $content = $content.TrimEnd() + [Environment]::NewLine +
            $replacement + [Environment]::NewLine
    }

    $utf8WithoutBom = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($EnvPath, $content, $utf8WithoutBom)
}

if (-not (Test-Path -Path $EnvPath -PathType Leaf)) {
    throw "Missing .env file: $EnvPath"
}

if (-not (Get-Command foundry -ErrorAction SilentlyContinue)) {
    throw "Foundry Local CLI was not found in PATH. Open a new PowerShell window after installing it."
}

Set-Location $ProjectRoot

Write-Host "[1/4] Starting Foundry Local server..." -ForegroundColor Cyan
$startOutput = & foundry server start 2>&1
$startExitCode = $LASTEXITCODE
$startOutput | ForEach-Object { Write-Host $_ }

if ($startExitCode -ne 0) {
    throw "Foundry Local server could not be started (exit code $startExitCode)."
}

Write-Host "[2/4] Discovering the active Foundry URL..." -ForegroundColor Cyan
$serviceUrl = $null

for ($attempt = 1; $attempt -le 15; $attempt++) {
    $statusText = (& foundry status 2>&1 | Out-String)
    $urlMatch = [Regex]::Match(
        $statusText,
        "http://127\.0\.0\.1:\d+"
    )

    if ($urlMatch.Success) {
        $serviceUrl = $urlMatch.Value
        break
    }

    Start-Sleep -Seconds 1
}

if (-not $serviceUrl) {
    throw "Foundry Local started, but its local service URL could not be discovered. Run 'foundry status' for details."
}

$baseUrl = "$serviceUrl/v1"
Set-DotEnvValue -Name "FOUNDRY_BASE_URL" -Value $baseUrl
Write-Host "Updated .env: FOUNDRY_BASE_URL=$baseUrl" -ForegroundColor Green

Write-Host "[3/4] Loading cached GPU models..." -ForegroundColor Cyan
& foundry model load $EmbeddingAlias

if ($LASTEXITCODE -ne 0) {
    throw "Could not load embedding model: $EmbeddingAlias"
}

& foundry model load $ChatAlias

if ($LASTEXITCODE -ne 0) {
    throw "Could not load chat model: $ChatAlias"
}

Write-Host "[4/4] Verifying the Foundry API and loaded models..." -ForegroundColor Cyan
$requestParameters = @{
    Uri = "$baseUrl/models"
    Method = "Get"
    TimeoutSec = 60
}
$modelResponse = Invoke-RestMethod @requestParameters

$loadedModelIds = @(
    $modelResponse.data |
        ForEach-Object { $_.id }
)

$expectedModelIds = @(
    (Get-DotEnvValue -Name "EMBEDDING_MODEL_ID"),
    (Get-DotEnvValue -Name "CHAT_MODEL_ID")
) | Where-Object { $_ }

$missingModelIds = @(
    $expectedModelIds |
        Where-Object { $_ -notin $loadedModelIds }
)

if ($missingModelIds.Count -gt 0) {
    throw "Foundry API is reachable, but expected model(s) are missing: $($missingModelIds -join ', ')"
}

Write-Host ""
Write-Host "GovRAG services are ready." -ForegroundColor Green
Write-Host "Foundry API: $baseUrl"
Write-Host "Loaded models:"
$loadedModelIds | ForEach-Object { Write-Host "- $_" }
Write-Host ""
Write-Host "Run GovRAG with:" -ForegroundColor Cyan
Write-Host 'python -u src\generation\rag_answer.py "When is a data protection impact assessment required under the GDPR?"'
