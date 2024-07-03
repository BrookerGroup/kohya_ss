import gradio as gr
import os
import argparse
from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.lora_gui_brookreator import lora_tab,train_model
from kohya_gui.custom_logging import setup_logging

def get_values(settings):
    values = []
    for setting in settings:
        if isinstance(setting, str):
            values.append(setting)
        else:
            values.append("" if setting.value is None else setting.value)
    return values

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

    interface = gr.Blocks(
        css="", title=f"Kohya_ss GUI", theme=gr.themes.Default()
    )

    # Assuming lora_tab returns the necessary settings
    with interface:
        result_tuple = lora_tab(headless=headless, config=config, use_shell_flag=use_shell_flag)
        settings_list = result_tuple[0]
        initial_values = get_values(settings_list)
        train_model(headless, False, *initial_values)

if __name__ == "__main__":
    # Set up hard-coded values
    config_path = "./config_brookreator.toml"
    debug = True
    headless = True
    do_not_use_shell = False

    # Set up logging
    log = setup_logging(debug=debug)

    run_training(config_path, headless, do_not_use_shell)
