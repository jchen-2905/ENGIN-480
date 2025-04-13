def gwa3_data_parser(filepath):
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]

    # Metadata on line 1 (index 1)
    meta = list(map(int, lines[1].split()))
    num_heights = meta[1]
    num_dirs = meta[2]

    # Line 2 = sector frequencies
    sector_freqs = list(map(float, lines[2].split()))

    # Line 3 = heights
    heights = list(map(float, lines[3].split()))

    data = {int(h): [] for h in heights}

    # Starting line for A/k values
    start = 4

    for dir_index in range(num_dirs):
        A_line = list(map(float, lines[start + dir_index * 2].split()))
        k_line = list(map(float, lines[start + dir_index * 2 + 1].split()))
        f = sector_freqs[dir_index] if dir_index < len(sector_freqs) else 1.0 / num_dirs

        for i, h in enumerate(heights):
            data[int(h)].append((f, A_line[i], k_line[i]))

    return data

def print_gwa3_data(data):
    for h in data:
        print(f"Height: {h} m")
        for i, (f, A, k) in enumerate(data[h]):
            print(f"  Dir {i*30}°: f={f:.3f}, A={A:.3f}, k={k:.3f}")

print("HornsRev1 data:")
print_gwa3_data(gwa3_data_parser("gwa3_HornsRev1_gwc.lib"))

print("AltaWind data:")
print_gwa3_data(gwa3_data_parser("gwa3_AltaWind_gwc.lib"))

print('Parsing complete')
