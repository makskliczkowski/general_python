with open("physics/thermal.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip() == "from functools import partial":
        lines[i] = ""
    elif len(line) > 80 and not line.startswith("def") and not "import" in line and not "#!" in line and not line.strip().startswith('"""'):
        # wrap lines or ignore them since black already ran
        pass
    elif line.startswith("#! End of file"):
        lines[i] = "# ! End of file\n"

with open("physics/thermal.py", "w") as f:
    f.writelines(lines)
