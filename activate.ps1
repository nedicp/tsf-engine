#activate.ps1
$envPath = (poetry env info --path)
& "$envPath\Scripts\activate.ps1"
Write-Host "✅ Activated DeepMark Poetry environment"
