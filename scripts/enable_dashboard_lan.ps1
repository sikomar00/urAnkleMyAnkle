# 이 스크립트는 반드시 "관리자 권한으로 실행"한 PowerShell에서 실행합니다.
# 대시보드가 사용하는 TCP 8052 포트만 Private(사내·가정) 네트워크에서 허용합니다.

$ruleName = "Predictive Maintenance Dashboard TCP 8052"

# 예전 규칙이 있으면 먼저 지워 중복 규칙을 만들지 않습니다.
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction SilentlyContinue

# 현재 대시보드 포트만 인바운드로 허용합니다.
New-NetFirewallRule `
    -DisplayName $ruleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 8052 `
    -Profile Private | Out-Null

Write-Host "방화벽 규칙이 등록되었습니다: TCP 8052 (Private 네트워크)"
Write-Host "같은 네트워크 접속 주소: http://192.168.219.110:8052/"
