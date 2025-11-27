import json
import csv

file = open("data/dataset.json","r")
dataset = json.load(file)
print(len(dataset))
print(dataset[0].keys())

out_file = open("data/input.csv",'w')
writer = csv.writer(out_file)

for transcript in dataset:
    conversation = transcript["conversation"]
    processed_conversation = {}
    for i in range(len(conversation)):
        processed_conversation[int(i)] = f"{conversation[i]["speaker"]}: {conversation[i]["text"]}"
    processed_conversation_json = json.dumps(processed_conversation, indent= 4)
    metadata = {}
    for info in transcript.items():
        if info[0] == "conversation":
            continue
        metadata[info[0]] = info[1]
    metadata_json = json.dumps(metadata, indent=4)
    writer.writerow([processed_conversation_json,metadata_json])


file.close()
out_file.close()
