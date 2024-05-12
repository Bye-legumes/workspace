import subprocess

def format_code_with_clang_format(file_path):
    # Specify the path to clang-format-12 if it's not the default version
    clang_format_command = ["clang-format", "-i", file_path]

    try:
        # Run the command
        subprocess.run(clang_format_command, check=True)
        print(f"Successfully formatted {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error formatting {file_path}: {e}")

# Example usage
format_code_with_clang_format("/workspace/workspace/ray/src/ray/raylet/local_task_manager.cc")
format_code_with_clang_format("/workspace/workspace/ray/src/ray/raylet/local_task_manager.h")
