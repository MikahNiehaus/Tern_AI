@echo off
echo Stopping whatever's running (base training or SFT) so you can game.
echo Progress is safe either way: checkpoints save every 50 iterations, run.bat picks back up automatically.
powershell -NoProfile -Command "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*train_gpt2_local.py*' -or $_.CommandLine -like '*train_sft.py*' }; if ($procs) { $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Write-Host 'Stopped, VRAM freed.' } else { Write-Host 'Nothing was running.' }"
pause
