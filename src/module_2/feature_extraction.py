import json
import csv

# parameters
threshold = 70
top_k = 5
classes = {
  "A1_Delivery_Issues": {
    "business_labels": ["BIS", "BIR", "BRV", "BTR", "BPU"],
    "conversational_markers":[]
  },

  "A2_Refund_Compensation_Issues": {
    "business_labels": ["BIS", "BTR", "BPU", "BRV"],
    "conversational_markers":[]
  },

  "A3_Payment_Billing_Challenges": {
    "business_labels": ["BIS", "BIR", "BPU", "BTR"],
    "conversational_markers":[]
  },

  "A4_Product_Quality_Issues": {
    "business_labels": ["BIS", "BIR"],
    "conversational_markers":[]
  },

  "A5_Checkout_OrderFlow_Confusion": {
    "business_labels": ["BIS", "BNW", "BRQ"],
    "conversational_markers":[]
  },

  "A6_General_Issue_Discovery": {
    "business_labels": ["BIS", "BNW"],
    "conversational_markers":[]
  },

  "B1_Agent_Behavior_Politeness_Professionalism": {
    "conversational_markers": ["POLI", "EMP", "NEG", "PRO", "OFF", "REP"],
    "business_labels": ["META"]
  },

  "B2_Positive_Agent_Sentiment": {
    "conversational_markers": ["POS", "THK", "AFF"],
    "business_labels": ["BFR"]
  },

  "B3_Agent_Handling_Quality": {
    "conversational_markers": ["INF", "QUE", "HES", "DEC", "EMP", "PRO"],
    "business_labels": ["BIR", "BRV"]
  },

  "C1_Subscription_Usage_Change": {
    "conversational_markers": ["CH1", "CH2", "CH3", "CH4", "CH5", "HES", "SEC", "NEG"],
    "business_labels": ["BPU"]
  },

  "C2_Fraud_Safety_Concerns": {
    "conversational_markers": ["FRU", "ANX", "DIS", "ESL"],
    "business_labels": ["BIS", "BPU"]
  },

  "C3_Product_Item_Demand_Signals": {
    "conversational_markers": ["INF", "QUE", "POS", "CLM"],
    "business_labels": ["BNW", "BRS"]
  },

  "C4_Persona_Specific_Issues": {
    "conversational_markers": ["FRU", "ANX", "NEG", "HES"],
    "business_labels": ["BNW"]
  },

  "E1_Repayment_Retention_Drivers": {
    "conversational_markers": ["HES", "DEC", "CH1", "CH2", "CH3", "CH4", "CH5"],
    "business_labels": ["BPU", "BTR", "BRV"]
  },

  "E2_Objection_Handling": {
    "conversational_markers": ["HES", "DIS", "FRU", "DEC"],
    "business_labels": ["BIR", "BPU"]
  }
}


# paths
# transcript_dict = json.loads(row[3])  # enter your path here
# preprocessed_data = [v for k, v in transcript_dict.items()]
# print(preprocessed_data)

# conversation = json.loads(row[0])  # enter your path
"""
turns = [
    {"turn": "Delivery was late", "conv_marker": "complaint", "business_label": "delivery"},
    {"turn": "Can you check my order?", "conv_marker": "question", "business_label": "delivery"},
    {"turn": "I want a refund", "conv_marker": "request", "business_label": "refund"},
] 
# this should be the format of turns to run flows function. """



# span
# def get_span_indices(turn_list, threshold, top_k):
#     # 1. Find indices where causal_score > threshold
#     above = [i for i, t in enumerate(turn_list) if t["causal_score"] > threshold]

#     # Case A: if some values exceed threshold → return them
#     if above:
#         return above

#     # Case B: none exceed threshold → take top_k highest
#     # Create list: (index, causal_score)
#     scored = [(i, t["causal_score"]) for i, t in enumerate(turn_list)]

#     # Sort by score descending and take top_k
#     top_k_indices = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#     # Return only the indices
#     indices = [i for i, score in top_k_indices]

#     lis = []
#     for i in indices:
#       lis.append(i-1) if i-1 >= 0 else None
#       lis.append(i)
#       lis.append(i+1)
    
#     span_index = list(lis)
#     # span_index = [str(i) for i in span_index]
#     return span_index

# def group_consecutive(nums):
#     if not nums:
#         return []

#     nums = sorted(set(nums))  # ensure sorted
#     groups = [[nums[0]]]

#     for n in nums[1:]:
#         # If current number continues the previous sequence (including duplicates)
#         if n == groups[-1][-1] or n == groups[-1][-1] + 1:
#             groups[-1].append(n)
#         else:
#             groups.append([n])
#     groups = [[str(x) for x in group] for group in groups]

#     return groups


# def get_span_turns(span_index, conversation):
#     span = []
#     for group in span_index:           # group is a list of indices
#         turns = [conversation[i] for i in group]
#         span.append(turns)
#     return span
def get_span_indices(turn_list, threshold, top_k):
    # 1. Find indices where causal_score > threshold
    above = [i for i, t in enumerate(turn_list) if t["causal_score"] > threshold]

    # Case A: if some values exceed threshold → return them
    if above:
        return above

    # Case B: none exceed threshold → take top_k highest
    # Create list: (index, causal_score)
    scored = [(i, t["causal_score"]) for i, t in enumerate(turn_list)]

    # Sort by score descending and take top_k
    top_k_indices = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    # Return only the indices
    indices = [i for i, score in top_k_indices]

    lis = []
    for i in indices:
      lis.append(i-1) if i-1 >= 0 else None
      lis.append(i)
      lis.append(i+1) if i+1 < len(turn_list) else None

    span_index = list(lis)
    return span_index

def get_span_turns(span_index, preprocessed):
    span = []
    for group in span_index:           # group is a list of indices
        turns = [{"turn_idx":i, "turn":preprocessed[i]["turn"],"causal_score": preprocessed[i]["causal_score"],"business_label": preprocessed[i]["business_label"], "conversational_marker":preprocessed[i]["conversational_marker"]} for i in group]
        span.append(turns)
    return span

def group_consecutive(nums):
    if not nums:
        return []

    nums = sorted(set(nums))  # ensure sorted
    groups = [[nums[0]]]

    for n in nums[1:]:
        # If current number continues the previous sequence (including duplicates)
        if n == groups[-1][-1] or n == groups[-1][-1] + 1:
            groups[-1].append(n)
        else:
            groups.append([n])
    groups = [[str(x) for x in group] for group in groups]

    return groups


# temporal
# def get_sentiment_vector(transcript_dict):

#     # Sort transcript keys numerically: "0","1","2",... → 0,1,2,…
#     sorted_items = sorted(transcript_dict.items(), key=lambda x: int(x[0]))

#     # Extract sentiment_score for each turn
#     sentiment_vector = [item["sentiment_score"] for _, item in sorted_items]

#     return sentiment_vector
def get_sentiment_vector(transcript_dict):

    # Sort transcript keys numerically: "0","1","2",... → 0,1,2,…
    sorted_items = sorted(transcript_dict.items(), key=lambda x: int(x[0]))

    # Extract sentiment_score for each turn
    sentiment_vector = [item["sentiment_score"] for _, item in sorted_items]

    return sentiment_vector


# flows
# def map_turns_to_classes(data, classes):
#     results = []

#     for i in json.loads(data["Preprocessed_Data"]):
#         # json.loads(data["Preprocessed_Data"])[i]['business_label']
        
#         # Always treat markers/labels as sets for easy checking
#         turn_markers = {json.loads(data["Preprocessed_Data"])[i]['conversational_marker']}
#         turn_blabels = {json.loads(data["Preprocessed_Data"])[i]['business_label']} 

#         matched_class = None

#         # check all classes
#         for class_name, rules in classes.items():
#             class_markers = set(rules["conv_markers"])
#             class_blabels = set(rules["business_labels"])

#             # Condition:
#             # marker ∈ class markers  AND  business ∈ class business labels
#             if turn_markers & class_markers and turn_blabels & class_blabels:
#                 matched_class = class_name
#                 break  # stop on first match (or remove break for multi-label mapping)

#         results.append({
#             "turn_idx": i,
#             "turn_text": json.loads(data["Preprocessed_Data"])[i]["turn"],
#             "turn_conv_marker": list(turn_markers),
#             "turn_business_label": list(turn_blabels),
#             "assigned_class": matched_class
#         })

#     return results
def map_classes_to_turns(turns_json, classes):
    turns = turns_json

    # Output container: class_name → list of matched turns
    class_to_turns = {cls: [] for cls in classes.keys()}

    for idx, turn in turns.items():

        # turn markers & labels
        turn_markers = {turn["conversational_marker"]}
        turn_blabels = {turn["business_label"]}

        # find which classes this turn belongs to
        for class_name, rules in classes.items():
            class_markers = set(rules["conversational_markers"])
            class_blabels = set(rules["business_labels"])

            # match condition
            if turn_markers & class_markers or turn_blabels & class_blabels:
                class_to_turns[class_name].append({
                    "turn_idx": idx,
                    "turn_text": turn["turn"],
                    "turn_conv_marker": list(turn_markers),
                    "turn_business_label": list(turn_blabels),
                    "causal_score": turn["causal_score"]
                })

    # remove classes that have zero matched turns
    class_to_turns = {cls: tlist for cls, tlist in class_to_turns.items() if tlist}

    return class_to_turns

# how to use fuctions

# span_index = get_span_indices(preprocessed_data, threshold=80, top_k=3)
# print(span_index)
# span_group = group_consecutive(span_index)
# span = get_span_turns(span_group)
# print(span)

# sent_vec = get_sentiment_vector(transcript_dict)
# print(sent_vec)

# mapped = map_turns_to_classes(turns, classes)

# input_csv = "/content/output (1).csv"               # <-- your existing CSV
# output_csv = "/content/output (1).csv"              # <-- overwrite same CSV

# rows = []

# with open(input_csv, "r", encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     transcripts_list = list(reader)

# for row in transcripts_list:

#     conv = json.loads(row["Transcript"]) if isinstance(row["Transcript"], str) else row["Transcript"]    # transcript ID from CSV

#     data = row  # lookup its data structure

#     # Compute span
#     span_idx = get_span_indices(list(json.loads(data["Preprocessed_Data"]).values()), threshold=80, top_k=3)
#     span_group = group_consecutive(span_idx)
#     span = get_span_turns(span_group, json.loads(data["Transcript"]))

#     # Compute sentiment vector
#     sent_vec = get_sentiment_vector(json.loads(data["Preprocessed_Data"]))

#     # Compute flows
#     mapped = map_turns_to_classes(json.loads(data["Preprocessed_Data"]), classes)

#     # Structure to store
#     info = {
#         "span": span,
#         "flows": mapped,
#         "temporal_vector": sent_vec
#     }

#     row["analysis"] = json.dumps(info)

#     rows.append(row)


# # Write CSV
# with open(output_csv # put transcript path here
#           , "w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=['Metadata', 'Inferred_Metadata', 'Preprocessed_Data', 'Transcript', 'analysis'])  # change column names
#     writer.writeheader()
#     writer.writerows(rows)
input_csv = "output (3).csv"               # <-- your existing CSV
output_csv = "output (3).csv"              # <-- overwrite same CSV

rows = []

with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    transcripts_list = list(reader)
i = 0
for row in transcripts_list:
    print(i)
    i+=1
    conv = json.loads(row["transcript"]) if isinstance(row["transcript"], str) else row["transcript"]    # transcript ID from CSV

    data = row  # lookup its data structure

    # Compute span
    span_idx = get_span_indices(list(json.loads(data["labels"]).values()), threshold=80, top_k=3)
    span_group = group_consecutive(span_idx)
    span = get_span_turns(span_group, json.loads(data["labels"]))

    # Compute sentiment vector
    sent_vec = get_sentiment_vector(json.loads(data["labels"]))

    # Compute flows
    mapped = map_classes_to_turns(json.loads(data["labels"]), classes)

    # Structure to store
    info = {
        "span": span,
        "flows": mapped,
        "temporal_vector": sent_vec
    }

    row["analysis"] = json.dumps(info)

    rows.append(row)


# Write CSV
with open(output_csv # put transcript path here
          , "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=['transcript', 'metadata', 'inferenced_metadata','labels', 'analysis'])  # change column names
    writer.writeheader()
    writer.writerows(rows)