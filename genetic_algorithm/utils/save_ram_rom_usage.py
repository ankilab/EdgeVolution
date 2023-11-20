import json


def save_ram_rom_usage(build_path: str, results_path: str):
    # get RAM usage
    with open(build_path + "/ram.json") as f:
        ram_usage = json.loads(f.read())["total_size"]

    # get ROM usage
    with open(build_path + "/rom.json") as f:
        rom_usage = json.loads(f.read())["total_size"]

    # load results from json
    with open(results_path) as f:
        results = json.loads(f.read())

    # add RAM usage to results
    results["ram_usage"] = ram_usage

    # add ROM usage to results
    results["rom_usage"] = rom_usage

    # save to results.json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)