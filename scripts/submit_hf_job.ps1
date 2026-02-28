param(
  [string]$Flavor = "a10g-small",
  [string]$Timeout = "6h",
  [string]$Config = "configs/grpo_config.yaml",
  [string]$Namespace = "",
  [string]$Repo = "",
  [switch]$Detach
)

$cmd = @("hf", "jobs", "uv", "run", "train.py")
if ($Detach) { $cmd += @("--detach") }

$cmd += @("--flavor", $Flavor)
if ($Namespace -ne "") { $cmd += @("--namespace", $Namespace) }
if ($Repo -ne "") { $cmd += @("--repo", $Repo) }

$cmd += @(
  "--timeout", $Timeout,
  "--secrets", "HF_TOKEN",
  "--secrets", "WANDB_API_KEY",
  "--secrets", "MISTRAL_API_KEY",
  "--env", "PYTHONUTF8=1",
  "--", "--config", $Config, "--ensure-system-deps"
)

Write-Host "Submitting HF Job:" ($cmd -join " ")
& $cmd[0] $cmd[1..($cmd.Length-1)]
exit $LASTEXITCODE
