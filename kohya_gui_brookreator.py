import gradio as gr
import os
import argparse
import boto3
from botocore.exceptions import NoCredentialsError
from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.lora_gui_brookreator import lora_tab, train_model
from kohya_gui.custom_logging import setup_logging

def get_values(settings):
    values = []
    for setting in settings:
        if isinstance(setting, str):
            values.append(setting)
        else:
            values.append("" if setting.value is None else setting.value)
    return values

def upload_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Create an S3 client
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except FileNotFoundError:
        log.error("The file was not found")
        return False
    except NoCredentialsError:
        log.error("Credentials not available")
        return False
    return True

def check_safetensor_then_save_to_s3():
    output_dir = "./outputs/"
    for file in os.listdir(output_dir):
        if file.endswith(".safetensors"):
            file_path = os.path.join(output_dir, file)
            log.info(f"Found .safetensor file: {file_path}")
            return file_path

def run_training(config_path, headless, do_not_use_shell):
    config = KohyaSSGUIConfig(config_file_path=config_path)
    if config.is_config_loaded():
        log.info(f"Loaded default GUI values from '{config_path}'...")

    use_shell_flag = True
    use_shell_flag = config.get("settings.use_shell", use_shell_flag)
    if do_not_use_shell:
        use_shell_flag = False

    if use_shell_flag:
        log.info("Using shell=True when running external commands...")

    with gr.Blocks():
        result_tuple = lora_tab(headless=headless, config=config, use_shell_flag=use_shell_flag)
        settings_list = result_tuple[0]
        initial_values = get_values(settings_list)
        train_model(headless, False, *initial_values)
        log.info("Training completed...")
        # exit(0)  # Ensure the script terminates

if __name__ == "__main__":
    # Set up hard-coded values
    config_path = "./config_brookreator.toml"
    debug = True
    headless = True
    do_not_use_shell = False

    # Set up logging
    log = setup_logging(debug=debug)

    run_training(config_path, headless, do_not_use_shell)

    log.info("Training completed. Checking for .safetensor file...")

    # Check for the .safetensor file in ./outputs/
    file_path = check_safetensor_then_save_to_s3()
    
    # Mock upload to S3
    if upload_to_s3(file_path, 'your-s3-bucket-name'):
        log.info(f"Successfully uploaded {file_path} to S3")
    else:
        log.error(f"Failed to upload {file_path} to S3")