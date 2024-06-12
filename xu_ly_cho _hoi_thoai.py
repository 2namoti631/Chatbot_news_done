import json
import csv

# Load JSON data
with open('data_basic_conver.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Open CSV file for writing
with open('data_conver.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Define CSV writer
    csv_writer = csv.writer(csv_file)

    # Write header
    csv_writer.writerow(['Pattern', 'Response','Tag'])

    # Iterate through intents
    for intent in data['intents']:
        # Iterate through patterns and responses
        for pattern, response in zip(intent['patterns'], intent['responses']):
            # Write tag, pattern, and response to CSV
            csv_writer.writerow([pattern, response, intent['tag']])

print("xử lý dữ liệu thành công. ")


