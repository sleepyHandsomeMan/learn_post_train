param(
    [ValidateSet("Smoke", "Diagnostic", "Formal")]
    [string]$Mode = "Smoke",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = "D:\Anaconda\envs\test3\python.exe"
$Entry = Join-Path $WorkspaceRoot "post_training_framework\scripts\run_grpo_train.py"
$Config = Join-Path $WorkspaceRoot "post_training_framework\configs\gsm8k_qwen3_0d6b_grpo_v7.json"

$Arguments = @($Entry, "--config", $Config)
switch ($Mode) {
    "Smoke" {
        $Arguments += @(
            "--train-file", (Join-Path $WorkspaceRoot "datasets\gsm8k_grpo\smoke_train_32.parquet"),
            "--total-training-steps", "3",
            "--eval-freq", "3",
            "--save-freq", "1",
            "--val-max-items", "20",
            "--run-name", "qwen3_0d6b_grpo_v7_preflight_smoke3",
            "--output-dir", (Join-Path $WorkspaceRoot "models\grpo\qwen3_0d6b_grpo_v7_preflight_smoke3")
        )
    }
    "Diagnostic" {
        $Arguments += @(
            "--total-training-steps", "20",
            "--eval-freq", "10",
            "--save-freq", "10",
            "--run-name", "qwen3_0d6b_grpo_v7_full5759_diag20",
            "--output-dir", (Join-Path $WorkspaceRoot "models\grpo\qwen3_0d6b_grpo_v7_full5759_diag20")
        )
    }
}

Write-Host "Mode: $Mode"
Write-Host "Command: $Python $($Arguments -join ' ')"
if (-not $DryRun) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "GRPO $Mode 执行失败，exit code=$LASTEXITCODE"
    }
}
