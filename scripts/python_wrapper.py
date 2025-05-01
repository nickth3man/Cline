import sys

def main():
    try:
        # Assuming the actual Python script to run is defined elsewhere or hardcoded
        script_to_run = "src/run_full_pipeline.py"
        args = sys.argv[1:]  # Get arguments passed to this wrapper
        full_command = ["python", script_to_run] + args
        command_str = " ".join(full_command)
        print(f"Running pipeline with correct Python interpreter: {command_str}")
        # Using exec to run the script (be cautious with exec)
        # Alternatively, you could use subprocess.run() for safer execution
        exec(open(script_to_run).read(), {"__file__": script_to_run, "__name__": "__main__"})
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()