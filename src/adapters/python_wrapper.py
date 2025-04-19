#!/usr/bin/env python
"""
Python Wrapper Script to fix import issues in the Cline project.
This script properly configures the Python path to allow both
absolute and relative imports to work correctly.
"""

import os
import sys
import subprocess
import importlib.util


def main():
    # Get the project root directory (where this script is located)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Add project root and src directory to sys.path
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

    # Check which Python interpreter we're using
    print(f"Using Python interpreter: {sys.executable}")

    # Check if we're using the correct virtual environment
    venv_path = os.path.join(project_root, "venv")
    venv_scripts = os.path.join(venv_path, "Scripts")
    venv_python = os.path.join(venv_scripts, "python.exe")

    if sys.executable.lower() != venv_python.lower():
        print(
            f"WARNING: Not using the project's virtual environment Python interpreter!"
        )
        print(f"Current interpreter: {sys.executable}")
        print(f"Expected interpreter: {venv_python}")

        # Re-run the command with the correct interpreter
        args = [venv_python] + sys.argv
        print(f"Restarting with correct interpreter: {' '.join(args)}")
        os.execv(venv_python, args)
        return

    # Parse command line arguments
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    # Store original argv and restore after setting up for the target module
    orig_argv = sys.argv.copy()

    # Run the appropriate command
    if command == "pipeline":
        # Set up arguments for the pipeline script
        sys.argv = ["run_full_pipeline.py"] + args

        try:
            print("Running pipeline with correct Python environment...")
            # Execute the run_full_pipeline.py script directly
            script_path = os.path.join(project_root, "src", "run_full_pipeline.py")
            with open(script_path, "r") as f:
                code = compile(f.read(), script_path, "exec")
                # Create a global namespace for the script with __name__ set to __main__
                globals_dict = {
                    "__name__": "__main__",
                    "__file__": script_path,
                }
                exec(code, globals_dict)
        except Exception as e:
            print(f"ERROR running pipeline: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    elif command == "verify":
        # Set up arguments for the verify script
        sys.argv = ["verify_output_structure.py"] + args

        try:
            print("Verifying output structure with correct Python environment...")
            # Execute the verify_output_structure.py script directly
            script_path = os.path.join(
                project_root, "src", "utils", "verify_output_structure.py"
            )
            with open(script_path, "r") as f:
                code = compile(f.read(), script_path, "exec")
                # Create a global namespace for the script with __name__ set to __main__
                globals_dict = {
                    "__name__": "__main__",
                    "__file__": script_path,
                }
                exec(code, globals_dict)
        except Exception as e:
            print(f"ERROR running verify: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    elif command == "run":
        # Run an arbitrary Python module from the project
        if not args:
            print("ERROR: Missing module path to run")
            print_usage()
            sys.exit(1)

        module_path = args[0]
        remaining_args = args[1:]

        # Set up arguments for the target module
        sys.argv = [module_path] + remaining_args

        try:
            print(f"Running module: {module_path}")
            if module_path.endswith(".py"):
                # Construct absolute path if necessary
                if not os.path.isabs(module_path):
                    module_path = os.path.join(project_root, module_path)

                # Run script directly
                with open(module_path, "r") as f:
                    code = compile(f.read(), module_path, "exec")
                    # Create a global namespace for the script with __name__ set to __main__
                    globals_dict = {
                        "__name__": "__main__",
                        "__file__": module_path,
                    }
                    exec(code, globals_dict)
            else:
                # Try to import and run as module
                module_name = module_path.replace("/", ".").replace("\\", ".")
                if module_name.endswith(".py"):
                    module_name = module_name[:-3]

                # Use importlib to execute the module
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    print(f"ERROR: Cannot find module: {module_name}")
                    sys.exit(1)

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Try to run main function if it exists
                if hasattr(module, "main"):
                    module.main()
        except Exception as e:
            print(f"ERROR running {module_path}: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)

    # Restore original argv
    sys.argv = orig_argv


def print_usage():
    print("Usage:")
    print("  python python_wrapper.py pipeline [args...]  - Run the full pipeline")
    print("  python python_wrapper.py verify [args...]    - Verify output structure")
    print("  python python_wrapper.py run <module> [args...] - Run a specific module")
    print("")
    print("Examples:")
    print("  python python_wrapper.py pipeline --help")
    print("  python python_wrapper.py verify")
    print("  python python_wrapper.py run src/transcription/transcription_workflow.py")


if __name__ == "__main__":
    main()
