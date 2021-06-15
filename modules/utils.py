"""Method utilities."""
from pathlib import Path


def open_csv(file_name, header):
    """Create or open csv file, and add header `header` in the former case."""
    # check if file exists
    if Path(file_name).is_file():
        # append to file and don't add header
        csv = open(file_name, 'a')
    else:
        # create file
        csv = open(file_name, 'w')
        # header
        csv.write(header + '\n')
    return csv
