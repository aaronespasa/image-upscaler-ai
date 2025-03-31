$RepoUrl = "https://github.com/ming053l/DRCT.git"
$FolderToCopy = "drct"
$BranchName = "main"
$TargetDir = "."
$TempDir = Join-Path $env:TEMP ([System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $TempDir | Out-Null

Write-Host "Cloning $RepoUrl into $TempDir..."
git clone --depth 1 --branch "$BranchName" "$RepoUrl" "$TempDir"

$SourceFolderPath = Join-Path $TempDir $FolderToCopy
if ($LASTEXITCODE -eq 0 -and (Test-Path $SourceFolderPath -PathType Container)) {
    Write-Host "Copying '$SourceFolderPath' to '$TargetDir'..."
    Copy-Item -Path $SourceFolderPath -Destination $TargetDir -Recurse -Force
    Write-Host "Folder copied successfully."
} else {
    Write-Host "Error: Failed to clone repository or folder '$FolderToCopy' not found in the repository." -ForegroundColor Red
}

Write-Host "Cleaning up temporary directory $TempDir..."
Remove-Item -Path $TempDir -Recurse -Force

Write-Host "Done."