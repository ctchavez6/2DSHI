import os
from experiment_set_up import user_input_validation as uiv
from constants import STEP_DESCRIPTIONS as sd

def step_nine(run_folder):
    step_description = sd.S09_DESC.value
    notes = uiv.yes_no_quit(step_description)

    if notes is True:
        notes = input("Write notes below:\n\n")
        notes_file = open(os.path.join(run_folder, 'notes.txt'), 'w+')
        notes_file.write(notes)
        notes_file.close()