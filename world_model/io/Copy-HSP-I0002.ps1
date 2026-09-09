[CmdletBinding()]
param(
    [string]$Source = "\\172.16.6.5\sleep\HSP\I0002",
    [string]$DestinationRoot = "I:\HSP",
    [ValidateRange(1, 128)]
    [int]$Workers = 32,
    [ValidateRange(0, 20)]
    [int]$Retries = 5,
    [ValidateRange(0, 300)]
    [int]$WaitSeconds = 2,
    [ValidateRange(1, 3600)]
    [int]$ReportSeconds = 5,
    [ValidateRange(0, 1000000)]
    [int]$SubjectLimit = 0
)

$ErrorActionPreference = "Stop"

function Format-Size {
    param([long]$Bytes)

    if ($Bytes -ge 1TB) { return "{0:N2} TB" -f ($Bytes / 1TB) }
    if ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
    if ($Bytes -ge 1MB) { return "{0:N2} MB" -f ($Bytes / 1MB) }
    if ($Bytes -ge 1KB) { return "{0:N2} KB" -f ($Bytes / 1KB) }
    return "$Bytes B"
}

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "This parallel copy script requires PowerShell 7 or later (pwsh.exe)."
}

$sourceRoot = $Source.TrimEnd("\")
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Source path does not exist or is not accessible: $sourceRoot"
}

$folderName = Split-Path -Leaf $sourceRoot
$destination = Join-Path $DestinationRoot $folderName
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$subjects = @(Get-ChildItem -LiteralPath $sourceRoot -Directory | Sort-Object Name)
if ($SubjectLimit -gt 0) {
    $subjects = @($subjects | Select-Object -First $SubjectLimit)
}
if ($subjects.Count -eq 0) {
    throw "No subject folders found under: $sourceRoot"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $DestinationRoot "copy_${folderName}_${timestamp}.log"
$failureLog = Join-Path $DestinationRoot "copy_${folderName}_${timestamp}_failures.tsv"

@(
    "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "Source: $sourceRoot"
    "Destination: $destination"
    "Workers: $Workers"
    "Subjects: $($subjects.Count)"
    "Excluded: .edf, .h5, and CSV filenames containing 'caisr'"
) | Out-File -LiteralPath $logFile -Encoding utf8

"source`tmessage" | Out-File -LiteralPath $failureLog -Encoding utf8

Write-Host "Source:      $sourceRoot"
Write-Host "Destination: $destination"
Write-Host "Subjects:    $($subjects.Count)"
Write-Host "Workers:     $Workers"
Write-Host "Excluded:    .edf, .h5, *caisr*.csv"
Write-Host "Log:         $logFile"
Write-Host "Starting parallel copy..."

$started = Get-Date
$lastReport = $started
$subjectsDone = 0
$filesFound = 0L
$copied = 0L
$skipped = 0L
$failed = 0L
$copiedBytes = 0L

$subjects |
    ForEach-Object -Parallel {
        $subject = $_
        $sourceRootLocal = $using:sourceRoot
        $destinationLocal = $using:destination
        $retryCount = $using:Retries
        $retryWaitSeconds = $using:WaitSeconds

        $subjectFiles = 0L
        $subjectCopied = 0L
        $subjectSkipped = 0L
        $subjectFailed = 0L
        $subjectCopiedBytes = 0L
        $failures = [System.Collections.Generic.List[string]]::new()

        try {
            $pendingDirectories = [System.Collections.Generic.Stack[string]]::new()
            $pendingDirectories.Push($subject.FullName)

            while ($pendingDirectories.Count -gt 0) {
                $currentSourceDir = $pendingDirectories.Pop()
                $relativeDir = [System.IO.Path]::GetRelativePath($sourceRootLocal, $currentSourceDir)
                $currentTargetDir = [System.IO.Path]::Combine($destinationLocal, $relativeDir)
                [System.IO.Directory]::CreateDirectory($currentTargetDir) | Out-Null

                foreach ($childDir in [System.IO.Directory]::EnumerateDirectories($currentSourceDir)) {
                    $pendingDirectories.Push($childDir)
                }

                foreach ($sourceFile in [System.IO.Directory]::EnumerateFiles($currentSourceDir)) {
                    $extension = [System.IO.Path]::GetExtension($sourceFile)
                    if ($extension.Equals(".edf", [System.StringComparison]::OrdinalIgnoreCase) -or
                        $extension.Equals(".h5", [System.StringComparison]::OrdinalIgnoreCase)) {
                        continue
                    }

                    $fileName = [System.IO.Path]::GetFileName($sourceFile)
                    if ($extension.Equals(".csv", [System.StringComparison]::OrdinalIgnoreCase) -and
                        $fileName.IndexOf("caisr", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                        continue
                    }

                    $subjectFiles++
                    $sourceInfo = [System.IO.FileInfo]::new($sourceFile)
                    $targetFile = [System.IO.Path]::Combine($currentTargetDir, $sourceInfo.Name)
                    $needsCopy = $true

                    if ([System.IO.File]::Exists($targetFile)) {
                        $targetInfo = [System.IO.FileInfo]::new($targetFile)
                        if ($targetInfo.Length -eq $sourceInfo.Length -and
                            $targetInfo.LastWriteTimeUtc -eq $sourceInfo.LastWriteTimeUtc) {
                            $needsCopy = $false
                        }
                    }

                    if (-not $needsCopy) {
                        $subjectSkipped++
                        continue
                    }

                    $copySucceeded = $false
                    $lastError = $null
                    for ($attempt = 0; $attempt -le $retryCount; $attempt++) {
                        try {
                            [System.IO.File]::Copy($sourceFile, $targetFile, $true)
                            [System.IO.File]::SetLastWriteTimeUtc($targetFile, $sourceInfo.LastWriteTimeUtc)

                            $targetInfo = [System.IO.FileInfo]::new($targetFile)
                            if ($targetInfo.Length -ne $sourceInfo.Length) {
                                throw "Size verification failed: source=$($sourceInfo.Length), target=$($targetInfo.Length)"
                            }

                            $copySucceeded = $true
                            break
                        }
                        catch {
                            $lastError = $_.Exception.Message
                            if ($attempt -lt $retryCount -and $retryWaitSeconds -gt 0) {
                                [System.Threading.Thread]::Sleep($retryWaitSeconds * 1000)
                            }
                        }
                    }

                    if ($copySucceeded) {
                        $subjectCopied++
                        $subjectCopiedBytes += $sourceInfo.Length
                    }
                    else {
                        $subjectFailed++
                        $safeMessage = ($lastError -replace "`r|`n|`t", " ")
                        $failures.Add("$sourceFile`t$safeMessage")
                    }
                }
            }
        }
        catch {
            $subjectFailed++
            $safeMessage = ($_.Exception.Message -replace "`r|`n|`t", " ")
            $failures.Add("$($subject.FullName)`t$safeMessage")
        }

        [pscustomobject]@{
            Subject = $subject.Name
            Files = $subjectFiles
            Copied = $subjectCopied
            Skipped = $subjectSkipped
            Failed = $subjectFailed
            CopiedBytes = $subjectCopiedBytes
            Failures = $failures.ToArray()
        }
    } -ThrottleLimit $Workers |
    ForEach-Object {
        $subjectsDone++
        $filesFound += $_.Files
        $copied += $_.Copied
        $skipped += $_.Skipped
        $failed += $_.Failed
        $copiedBytes += $_.CopiedBytes

        if ($_.Failures.Count -gt 0) {
            $_.Failures | Out-File -LiteralPath $failureLog -Encoding utf8 -Append
        }

        $now = Get-Date
        if (($now - $lastReport).TotalSeconds -ge $ReportSeconds -or $subjectsDone -eq $subjects.Count) {
            $elapsedSeconds = [Math]::Max(($now - $started).TotalSeconds, 0.001)
            $speed = [long]($copiedBytes / $elapsedSeconds)
            $percent = [Math]::Round(($subjectsDone / $subjects.Count) * 100, 2)
            Write-Host ("[{0}] {1}% | subjects {2}/{3} | copied {4}, skipped {5}, failed {6} | {7} copied | avg {8}/s" -f `
                (Get-Date -Format "HH:mm:ss"), $percent, $subjectsDone, $subjects.Count,
                $copied, $skipped, $failed, (Format-Size $copiedBytes), (Format-Size $speed))
            $lastReport = $now
        }
    }

$finished = Get-Date
$summary = @(
    "Finished: $($finished.ToString('yyyy-MM-dd HH:mm:ss'))"
    "Elapsed: $($finished - $started)"
    "Subjects: $subjectsDone"
    "Eligible files: $filesFound"
    "Copied: $copied"
    "Skipped: $skipped"
    "Failed: $failed"
    "Copied bytes: $copiedBytes"
    "Failure log: $failureLog"
)
$summary | Out-File -LiteralPath $logFile -Encoding utf8 -Append

Write-Host "Finished."
Write-Host "Subjects: $subjectsDone"
Write-Host "Eligible files: $filesFound"
Write-Host "Copied:  $copied"
Write-Host "Skipped: $skipped"
Write-Host "Failed:  $failed"
Write-Host "Copied this run: $(Format-Size $copiedBytes)"
Write-Host "Log:     $logFile"
Write-Host "Failures: $failureLog"

if ($failed -gt 0) {
    exit 1
}

exit 0
