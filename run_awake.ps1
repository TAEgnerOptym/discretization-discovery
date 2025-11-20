# run_awake.ps1  (ASCII-safe)

$python = "C:\Users\Xiaohang\AppData\Local\Programs\Python\Python312\python.exe"
$script = "C:\Julian_harmony\discretization-discovery\main_simple.py"

Write-Host "Preventing system sleep while running..."
powercfg -change -standby-timeout-ac 0
powercfg -change -monitor-timeout-ac 0

try {
    & $python $script
    Write-Host "Run completed."
}
catch {
    Write-Host "Error: $($_.Exception.Message)"
}
finally {
    Write-Host "Restoring sleep settings..."
    powercfg -change -standby-timeout-ac 15
    powercfg -change -monitor-timeout-ac 5
    Write-Host "Sleep settings restored."
}
