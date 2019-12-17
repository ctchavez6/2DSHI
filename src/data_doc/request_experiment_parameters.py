import re

def can_string_be_converted_to_float(element):
    if re.match(r'^-?\d+(?:\.\d+)?$', element) is None:
        return False
    return True

def does_string_end_with_number(string):
    m = re.search(r'\d+$', string)
    # if the string ends in digits m will be a Match object, or None otherwise.
    if m is not None:
        return True
    return False

def can_string_be_converted_to_float(element):
    if re.match(r'^-?\d+(?:\.\d+)?$', element) is None:
        return False
    return True

def get_crystal_temperature(crystal_number):
    user_input = input("Input Crystal {} Temperature (Celsius): ".format(crystal_number))
    while not can_string_be_converted_to_float(user_input) or not does_string_end_with_number(user_input):
        user_input = input("Input Crystal {} Temperature (Celsius): ".format(crystal_number))
    temperature = float(user_input)
    return temperature

def get_target_description():
    targets = {"T1": "Wire", "T2": "Cross-Hair", "T3": "TBD 3", "T4": "TBD 4", "T5": "TBD 5"}
    target_description_string = ""
    for key in targets:
        target_description_string += "{}: {}\n".format(key, targets[key])
    user_input = input("{}Input Target Descriptor (See options above): ".format(target_description_string))
    if does_string_end_with_number(user_input):
        user_input = "T" + str(user_input)
    while user_input.upper() not in targets.keys():
        if does_string_end_with_number(user_input):
            user_input = "T" + str(user_input)
        else:
            user_input = input("{}Input Target Descriptor (See options above): ".format(target_description_string))
    return targets[user_input.upper()]

def get_compensator_angle():
    return float(input("Input compensator angle: "))

