import os

def step_ten(run_folder):
    notes = input("Step 10 - Write some notes to a file? - Proceed? (y/n): ")

    if notes.lower() == 'y':
        notes = input("Write notes below:\n\n")
        notes_file = open(os.path.join(run_folder, 'notes.txt'), 'w+')
        notes_file.write(notes)
        notes_file.close()