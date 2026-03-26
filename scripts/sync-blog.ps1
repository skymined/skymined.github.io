param(
  [string]$SourceRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [string]$TargetRepo = (Join-Path (Split-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path -Parent) "blog"),
  [string]$TargetRemote = "https://github.com/skymined/blog.git",
  [switch]$SkipPush,
  [switch]$SkipPull,
  [switch]$VerifyBuild
)

$ErrorActionPreference = "Stop"

if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $false
}

function Write-Step {
  param([string]$Message)
  Write-Host "[blog-sync] $Message"
}

function ConvertTo-GitArgumentString {
  param([string[]]$GitArgs)

  $quoted = foreach ($arg in $GitArgs) {
    if ($arg -notmatch '[\s"]') {
      $arg
      continue
    }

    '"' + ($arg -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
  }

  return ($quoted -join " ")
}

function Invoke-GitInDirectory {
  param(
    [string]$WorkingDirectory,
    [string[]]$GitArgs
  )

  $processInfo = New-Object System.Diagnostics.ProcessStartInfo
  $processInfo.FileName = "git"
  $processInfo.WorkingDirectory = $WorkingDirectory
  $processInfo.UseShellExecute = $false
  $processInfo.CreateNoWindow = $true
  $processInfo.RedirectStandardOutput = $true
  $processInfo.RedirectStandardError = $true
  $processInfo.Arguments = ConvertTo-GitArgumentString -GitArgs $GitArgs

  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $processInfo
  $null = $process.Start()

  $stdout = $process.StandardOutput.ReadToEnd()
  $stderr = $process.StandardError.ReadToEnd()
  $process.WaitForExit()

  $output = ($stdout, $stderr -join "").Trim()
  if ($process.ExitCode -ne 0) {
    throw "git $($GitArgs -join ' ') failed in $WorkingDirectory`n$output"
  }

  return $output
}

function Invoke-Git {
  param(
    [string]$Repo,
    [string[]]$GitArgs
  )

  return Invoke-GitInDirectory -WorkingDirectory $Repo -GitArgs $GitArgs
}

function Invoke-Robocopy {
  param(
    [string]$From,
    [string]$To,
    [string[]]$Options
  )

  & robocopy $From $To @Options | Out-Null
  $exitCode = $LASTEXITCODE
  if ($exitCode -ge 8) {
    throw "robocopy failed with exit code $exitCode"
  }
}

if (-not (Test-Path (Join-Path $SourceRepo ".git"))) {
  throw "Source repo not found: $SourceRepo"
}

if (-not (Test-Path $TargetRepo)) {
  Write-Step "Cloning blog repo to $TargetRepo"
  Invoke-GitInDirectory -WorkingDirectory (Split-Path $TargetRepo -Parent) -GitArgs @("clone", $TargetRemote, $TargetRepo) | Out-Null
}

if (-not (Test-Path (Join-Path $TargetRepo ".git"))) {
  throw "Target repo is not a git repo: $TargetRepo"
}

$sourceTempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("blog-sync-" + [System.Guid]::NewGuid().ToString("N"))
$sourceSnapshot = Join-Path $sourceTempRoot "snapshot"
$sourceArchive = Join-Path $sourceTempRoot "source.zip"
New-Item -ItemType Directory -Path $sourceSnapshot -Force | Out-Null

try {
  Write-Step "Exporting source HEAD snapshot"
  Invoke-Git -Repo $SourceRepo -GitArgs @("archive", "--format=zip", "-o", $sourceArchive, "HEAD") | Out-Null
  Expand-Archive -Path $sourceArchive -DestinationPath $sourceSnapshot -Force

$targetStatus = Invoke-Git -Repo $TargetRepo -GitArgs @("status", "--porcelain")
if ($targetStatus) {
  throw "Target repo has local changes. Please clean $TargetRepo before syncing.`n$targetStatus"
}

if (-not $SkipPull) {
  Write-Step "Updating target repo"
  Invoke-Git -Repo $TargetRepo -GitArgs @("pull", "--ff-only", "origin", "main") | Out-Null
}

Write-Step "Mirroring source into blog repo"
Invoke-Robocopy -From $sourceSnapshot -To $TargetRepo -Options @(
  "/MIR",
  "/XD",
  ".git",
  ".github",
  ".githooks",
  ".obsidian",
  ".quartz-cache",
  "docs",
  "node_modules",
  "private",
  "public",
  "redirect-site",
  "scripts",
  "/NJH",
  "/NJS",
  "/NFL",
  "/NDL",
  "/NP"
)

$configPath = Join-Path $TargetRepo "quartz.config.ts"
$configText = [System.IO.File]::ReadAllText($configPath)
$baseUrlRegex = [System.Text.RegularExpressions.Regex]::new('baseUrl:\s*"[^"]+"')
$updatedConfig = $baseUrlRegex.Replace($configText, 'baseUrl: "skymined.github.io/blog"', 1)

if ($configText -ne $updatedConfig) {
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($configPath, $updatedConfig, $utf8NoBom)
}

if ($VerifyBuild) {
  Write-Step "Installing dependencies for verification"
  Push-Location $TargetRepo
  try {
    & npm ci
    if ($LASTEXITCODE -ne 0) {
      throw "npm ci failed while verifying the blog repo"
    }

    Write-Step "Building Quartz for verification"
    & npx quartz build
    if ($LASTEXITCODE -ne 0) {
      throw "npx quartz build failed while verifying the blog repo"
    }
  } finally {
    Pop-Location
  }
}

$pendingChanges = Invoke-Git -Repo $TargetRepo -GitArgs @("status", "--porcelain")
if (-not $pendingChanges) {
  Write-Step "No blog changes to sync"
  exit 0
}

$sourceSha = Invoke-Git -Repo $SourceRepo -GitArgs @("rev-parse", "--short", "HEAD")
$sourceMessage = Invoke-Git -Repo $SourceRepo -GitArgs @("log", "-1", "--pretty=%s")
$commitMessage = "sync from skymined.github.io: $sourceSha $sourceMessage"

Write-Step "Committing synced changes"
Invoke-Git -Repo $TargetRepo -GitArgs @("add", "-A") | Out-Null
Invoke-Git -Repo $TargetRepo -GitArgs @("commit", "-m", $commitMessage) | Out-Null

if ($SkipPush) {
  Write-Step "Skipping push"
  exit 0
}

Write-Step "Pushing blog repo"
Invoke-Git -Repo $TargetRepo -GitArgs @("push", "origin", "main") | Out-Null
Write-Step "Blog sync completed"
} finally {
  if (Test-Path $sourceTempRoot) {
    Remove-Item -LiteralPath $sourceTempRoot -Recurse -Force
  }
}
