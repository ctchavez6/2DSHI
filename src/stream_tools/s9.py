import os
from experiment_set_up import user_input_validation as uiv

def step_nine(run_folder):
    step_description = "Step 9 - Write some notes to a file?"
    notes = uiv.yes_no_quit(step_description)

    if notes is True:
        notes = input("Write notes below:\n\n")
        notes_file = open(os.path.join(run_folder, 'notes.txt'), 'w+')
        notes_file.write(notes)
        notes_file.close()