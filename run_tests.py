import subprocess

# List of command line arguments to run with main.py
commands = [
    # Use attention layer at the end (KAF used)
    ["--graph_type", "line", "--attention_layer", "True", "--activation", "kaf"],
    ["--graph_type", "cycle", "--attention_layer", "True", "--activation", "kaf"],
    # Apply KAF with random initialization (no attention)
    ["--graph_type", "line", "--attention_layer", "False", "--activation", "random_kaf"],
    ["--graph_type", "cycle", "--attention_layer", "False", "--activation", "random_kaf"],
    # Apply KAF with ELU mimic strategy (no attention)
    ["--graph_type", "line", "--attention_layer", "False", "--activation", "kaf"],
    ["--graph_type", "cycle", "--attention_layer", "False", "--activation", "kaf"],
    # Original - ReLU activation and no attention
    ["--graph_type", "line", "--attention_layer", "False", "--activation", "relu"],
    ["--graph_type", "cycle", "--attention_layer", "False", "--activation", "relu"]
]

with open("results.txt", "a") as file:
    print("Commands:")
    for cmd in commands:
        full_command = ["python", "main.py"] + cmd
        print(f"{' '.join(full_command)}")

        # result = subprocess.run(full_command, capture_output=True, text=True)
        
        # file.write(f"Command: {' '.join(full_command)}\n")
        # file.write("Output:\n")
        # file.write(result.stdout)
        # file.write("\nError (if any):\n")
        # file.write(result.stderr if result.stderr else "No errors.\n")
        # file.write("=" * 40 + "\n")
        # print("Done")