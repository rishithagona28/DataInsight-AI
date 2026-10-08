' Starts DataInsight AI in the background (no command window) and opens it in your browser.
' If DataInsight AI is already running, it just opens the page.
Option Explicit
Const URL = "http://localhost:8501/"
Dim sh, fso, appDir, py, q, i
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
appDir = fso.GetParentFolderName(WScript.ScriptFullName)
py = fso.GetAbsolutePathName(appDir & "\..\venv\Scripts\python.exe")
q = Chr(34)

Function IsRunning()
    Dim http
    IsRunning = False
    On Error Resume Next
    Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
    http.setTimeouts 1000, 1000, 1000, 1000
    http.Open "GET", URL, False
    http.Send
    If Err.Number = 0 Then
        If http.Status = 200 Then IsRunning = True
    End If
    Err.Clear
    On Error GoTo 0
End Function

' Stops the app's server AND every helper process it started (e.g. model-training
' workers); otherwise leftovers can keep the port blocked.
Sub FreePort()
    CreateObject("WScript.Shell").Run "powershell -NoProfile -WindowStyle Hidden -Command ""$ids = @(Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { $_.OwningProcess } | Sort-Object -Unique); foreach ($id in $ids) { taskkill /PID $id /T /F | Out-Null; Get-CimInstance Win32_Process -Filter ('ParentProcessId=' + $id) | ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null } }""", 0, True
End Sub

If Not IsRunning() Then
    FreePort  ' clear any leftover process still holding the port
    If Not fso.FileExists(py) Then
        MsgBox "Can't find Python at:" & vbCrLf & py, vbExclamation, "DataInsight AI"
        WScript.Quit 1
    End If
    sh.CurrentDirectory = appDir
    ' window style 0 = hidden; output goes to run_log.txt for troubleshooting
    sh.Run "cmd /c " & q & q & py & q & " -m streamlit run app.py --server.port 8501 --server.headless true > run_log.txt 2>&1" & q, 0, False
    For i = 1 To 60
        WScript.Sleep 500
        If IsRunning() Then Exit For
    Next
    If Not IsRunning() Then
        MsgBox "DataInsight AI didn't start. Details are in run_log.txt in:" & vbCrLf & appDir, vbExclamation, "DataInsight AI"
        WScript.Quit 1
    End If
End If
sh.Run URL
