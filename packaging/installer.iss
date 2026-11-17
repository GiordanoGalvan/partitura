; Inno Setup: installatore Windows di Partitura
#define AppVer GetEnv("PARTITURA_VERSIONE")
[Setup]
AppId={{6E2B8C1E-5A44-4F3B-9C1E-7D2A0B1F4C31}
AppName=Partitura
AppVersion={#AppVer}
AppPublisher=madfi
DefaultDirName={autopf}\Partitura
DefaultGroupName=Partitura
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=Partitura-Setup-{#AppVer}
SetupIconFile=partitura.ico
UninstallDisplayIcon={app}\Partitura.exe
Compression=lzma2/max
SolidCompression=yes
DiskSpanning=no
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktop"; Description: "Crea un'icona sul desktop"

[Files]
Source: "..\dist\Partitura\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Partitura"; Filename: "{app}\Partitura.exe"
Name: "{group}\Disinstalla Partitura"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Partitura"; Filename: "{app}\Partitura.exe"; Tasks: desktop

[Run]
Filename: "{app}\Partitura.exe"; Description: "Avvia Partitura"; Flags: nowait postinstall skipifsilent
