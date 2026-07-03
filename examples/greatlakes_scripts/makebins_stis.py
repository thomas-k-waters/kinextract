import os

# Sequence of the number of entries per bin
sequence = [143, 142, 89, 55, 34, 21, 13, 8, 3, 2, 1, 1, 1, 2, 3, 8, 13, 21, 34, 55, 89, 142, 143]

# Total number of spec entries to generate
total_entries = sum(sequence)
start_spec = 512

# Create directories to store the files if they don't already exist
# os.makedirs("bins", exist_ok=True)

# Create bin files for the central, right, and left sides
def create_bin_file(bin_name, entry_specs):
    with open(f"{bin_name}", "w") as bin_file:
        for spec in entry_specs:
            bin_file.write(f"spec_{spec:04d}\n")

# Generate bins on the right side
current_spec = start_spec + 1
for i, num_entries in enumerate(sequence[12:], start=2):
    bin_name = f"bin{i:02d}01"
    entry_specs = [(current_spec + j) % total_entries for j in range(num_entries)]
    create_bin_file(bin_name, entry_specs)
    current_spec += num_entries

# Central bin
create_bin_file("bin0105", [start_spec])

# Generate bins on the left side
current_spec = start_spec - 1
for i, num_entries in enumerate(sequence[11::-1], start=2):
    bin_name = f"bnn{i:02d}01"
    entry_specs = [(current_spec - j) % total_entries for j in range(num_entries)]
    create_bin_file(bin_name, entry_specs)
    current_spec -= num_entries
