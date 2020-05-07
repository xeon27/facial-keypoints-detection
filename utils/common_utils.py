import os


def read_file(file_path, headers=False):
    """ Read csv file using python """
    data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            data.append(line.strip().split(","))
        
    # Remove line containing headers
    if headers:
        data = data[1:]
        
    return data