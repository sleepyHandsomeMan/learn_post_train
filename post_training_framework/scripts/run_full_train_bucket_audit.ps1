$ErrorActionPreference = "Stop"

# 全量训练集分桶审计:
# 1. SFT greedy@1 跑完整 train
# 2. SFT oracle@8 跑完整 train
# 3. 根据 greedy/oracle 结果输出分桶统计和每个桶的 parquet

$Python = "d:\Anaconda\envs\test3\python.exe"
$Config = ".\post_training_framework\configs\gsm8k_qwen3_0d6b.json"
$SftAdapter = ".\models\sft\qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2"
$TrainFile = "datasets/gsm8k_sft/train.parquet"
$MaxItems = "7473"

$GreedyBatchSize = "64"
$OracleBatchSize = "16"

$GreedyRunName = "0d6b_eosfix2_greedy_train_full7473_len256_bs64"
$GreedyDir = ".\eval_results\sft_model\$GreedyRunName"
$GreedyJsonl = "$GreedyDir\$($GreedyRunName)_eval_7473_max256_full.jsonl"

$OracleRunName = "0d6b_eosfix2_oracle8_train_full7473_len256_temp0d7_bs16"
$OracleDir = ".\eval_results\sft_model\$OracleRunName"
$OracleJsonl = "$OracleDir\$($OracleRunName).jsonl"

$BucketDir = ".\datasets\gsm8k_grpo\audits\buckets_train_full7473_sft_eosfix2_oracle8_len256_bs64_bs16"

function Invoke-Step {
    param(
        [string] $Name,
        [scriptblock] $Body
    )
    Write-Output "===== START $Name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
    & $Body
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Output "===== END $Name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
}

if (-not (Test-Path $GreedyJsonl)) {
    Invoke-Step "sft_greedy_full_train" {
        & $Python .\post_training_framework\scripts\run_sft_eval.py `
            --config $Config `
            --adapter-dir $SftAdapter `
            --max-new-tokens 256 `
            --max-items $MaxItems `
            --eval-batch-size $GreedyBatchSize `
            --run-name $GreedyRunName `
            --output-dir $GreedyDir `
            --set "dataset.eval_file=$TrainFile"
    }
}
else {
    Write-Output "SKIP sft_greedy_full_train: $GreedyJsonl exists"
}

if (-not (Test-Path $OracleJsonl)) {
    Invoke-Step "sft_oracle8_full_train" {
        & $Python .\post_training_framework\scripts\run_oracle_eval.py `
            --config $Config `
            --model-kind sft `
            --adapter-dir $SftAdapter `
            --max-new-tokens 256 `
            --max-items $MaxItems `
            --oracle-k 8 `
            --eval-batch-size $OracleBatchSize `
            --temperature 0.7 `
            --top-p 1.0 `
            --top-k 50 `
            --run-name $OracleRunName `
            --output-dir $OracleDir `
            --set "dataset.eval_file=$TrainFile"
    }
}
else {
    Write-Output "SKIP sft_oracle8_full_train: $OracleJsonl exists"
}

Invoke-Step "build_bucket_dataset" {
    & $Python .\post_training_framework\scripts\build_grpo_bucket_dataset.py `
        --source-parquet .\datasets\gsm8k_sft\train.parquet `
        --greedy-jsonl $GreedyJsonl `
        --oracle-jsonl $OracleJsonl `
        --output-dir $BucketDir
}
