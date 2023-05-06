import os
import json
import pandas as pd
from datetime import datetime

# Get the directory of the current script
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Create an absolute path to the data directory
DATA_DIRECTORY = os.path.join(CURRENT_DIRECTORY, '..', 'data')

# Create an absolute path to the intents directory
INTENTS_DIRECTORY = os.path.join(DATA_DIRECTORY, 'intents')

# Create an absolute path to the intents file
INTENTS_PATH = os.path.join(INTENTS_DIRECTORY, 'intents.json')


def excel2json(file_path):
    # Load the excel file
    excel_file = pd.read_excel(file_path, sheet_name=None)

    # Create a new dictionary to store the intents
    intents = {"intents": []}

    # Iterate over the sheets in the excel file
    for sheet_name, sheet_data in excel_file.items():
        # Iterate over the rows in the sheet
        for _, row in sheet_data.iterrows():
            # Check if the row has a tag value
            if pd.notnull(row['tag']):
                # Create a new intent
                intent = {
                    'tag': row['tag'],
                    'patterns': [],
                    'response': row['response']
                }

                # Append the pattern to the intent
                intent['patterns'].append(row['patterns'])

                # Append the new intent to the intents dictionary
                intents['intents'].append(intent)
            else:
                # If the row doesn't have a tag value, append the pattern to the last added intent
                if len(intents['intents']) > 0:
                    last_intent = intents['intents'][-1]
                    last_intent['patterns'].append(row['patterns'])

    # Return the intents dictionary
    return intents


def create_tag_response_json(intents_data, output_file_path):
    # Create a new list to store the intents
    intents = []

    # Iterate over the intents in the input JSON file
    for intent_data in intents_data['intents']:
        # Create a new intent dictionary with only the tag and response
        intent = {
            'tag': intent_data['tag'],
            'response': intent_data['response']
        }

        # Append the new intent to the intents list
        intents.append(intent)

    # Create a new dictionary with the intents list
    result = {'intents': intents,
              'date': datetime.now().strftime("%B %d, %Y %I:%M %p")}

    # Write the result dictionary to a new JSON file
    with open(output_file_path, 'w') as f:
        json.dump(result, f, indent=4)


def load_training_data():
    return excel2json(os.path.join(DATA_DIRECTORY, 'chatplm_brain.xlsx'))


def load_data():
    with open(INTENTS_PATH) as file:
        intent_dict = json.load(file)
        print(f'Loaded: {INTENTS_PATH}')
        return intent_dict
