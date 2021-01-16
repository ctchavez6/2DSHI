import os


def get_latest_run():
    """
    TODO add documentation.
    """
    data_directory = os.path.join("D:", "")

    all_runs = sorted([os.path.join(data_directory, path) for path in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, path))
                       and path not in ["$RECYCLE.BIN", "System Volume Information"]])


    if len(all_runs) < 1:
        return None
    else:
        return all_runs[-1]

def get_all_runs():
    """
    TODO add documentation.
    """
    data_directory = os.path.join("D:", "")

    all_runs = sorted([os.path.join(data_directory, path) for path in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, path))
                       and path not in ["$RECYCLE.BIN", "System Volume Information"]])


    if len(all_runs) < 1:
        return None
    else:
        return all_runs