#define AppName "Xinbao"
#ifndef AppVersion
#define AppVersion "1.0.0"
#endif
#define AppPublisher "Xinbao"
#define PayloadDir "..\dist\Xinbao"

[Setup]
AppId={{A6B4B14E-2B14-4B7A-9E0D-5A4B20F7A7B1}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Xinbao
DefaultGroupName={#AppName}
PrivilegesRequired=admin
OutputDir=..\dist\installer
OutputBaseFilename=Xinbao-Setup-v{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\Xinbao.exe

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autodesktop}\芯宝 Xinbao"; Filename: "{app}\Xinbao.exe"; WorkingDir: "{app}"
Name: "{group}\芯宝 Xinbao"; Filename: "{app}\Xinbao.exe"; WorkingDir: "{app}"

[Run]
Filename: "{app}\Xinbao.exe"; Description: "立即启动芯宝"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/IM Xinbao.exe /F"; Flags: runhidden

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
