' Stops Ana (whatever is serving http://localhost:8501).
' Stops the app's server AND every helper process it started (e.g. model-training
' workers); otherwise leftovers can keep the port blocked.
Sub FreePort()
    CreateObject("WScript.Shell").Run "powershell -NoProfile -WindowStyle Hidden -Command ""$ids = @(Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { $_.OwningProcess } | Sort-Object -Unique); foreach ($id in $ids) { taskkill /PID $id /T /F | Out-Null; Get-CimInstance Win32_Process -Filter ('ParentProcessId=' + $id) | ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null } }""", 0, True
End Sub

FreePort
MsgBox "Ana has been stopped.", vbInformation, "Ana"
