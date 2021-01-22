import os


def get_latest_run():
    """
    This function returns the directory of the latest run by using alphanumerical sorting, if no runs return None.

    Returns:
        str: Name of the most recent run in YYYY_MM_DD_HH_mm format (MM: Two Digit Month, mm: Two Digit Minute).
        None: If hardcoded has no saved runs.

    TODO: Change Hardcoded Data Directory from Hard Coded to programmable.
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