# Check if a virtual environment is active and deactivate it if necessary
if ($env:VIRTUAL_ENV) {
    # Write-Host "Deactivating the virtual environment to test for modules installed locally..."
    & deactivate
}

# Activate the virtual environment
# Write-Host "Activating the virtual environment..."
& .\venv\Scripts\activate
$env:PATH += ";$($MyInvocation.MyCommand.Path)\venv\Lib\site-packages\torch\lib"

# Debug info about system
# python.exe .\setup\debug_info.py

# Validate the requirements and store the exit code
python.exe .\setup\validate_requirements.py

# Check the exit code and stop execution if it is not 0
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to validate requirements. Exiting script..."
    exit $LASTEXITCODE
}

# If the exit code is 0, read arguments from gui_parameters.txt (if it exists)
# and run the kohya_gui.py script with the command-line arguments
if ($LASTEXITCODE -eq 0) {
    # Write-Host "The arguments passed to this script were: $args_combo"
    python.exe kohya_gui_brookreator.py
}
