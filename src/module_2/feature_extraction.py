import json
import csv

# parameters
threshold = 70
top_k = 5
# classes = pass

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
      lis.append(i+1)
    
    span_index = list(lis)
    # span_index = [str(i) for i in span_index]
    return span_index

def group_consecutive(span_index):
    nums = sorted(set(span_index))  # remove duplicates + sort
    groups = []
    current_group = [nums[0]]

    for n in nums[1:]:
        if n == current_group[-1] + 1:
            current_group.append(n)
        else:
            groups.append(current_group)
            current_group = [n]

    groups.append(current_group)
    return groups


def get_span_turns(span_index, conversation):
    span = []
    for group in span_index:           # group is a list of indices
        turns = [conversation[i] for i in group]
        span.append(turns)
    return span


# temporal
def get_sentiment_vector(transcript_dict):

    # Sort transcript keys numerically: "0","1","2",... → 0,1,2,…
    sorted_items = sorted(transcript_dict.items(), key=lambda x: int(x[0]))

    # Extract sentiment_score for each turn
    sentiment_vector = [item["sentiment_score"] for _, item in sorted_items]

    return sentiment_vector


# flows
def map_turns_to_classes(turns, classes):
    results = []

    for i, turn in enumerate(turns):

        # Always treat markers/labels as sets for easy checking
        turn_markers = set(turn["conv_marker"]) if isinstance(turn["conv_marker"], list) else {turn["conv_marker"]}
        turn_blabels = set(turn["business_label"]) if isinstance(turn["business_label"], list) else {turn["business_label"]}

        matched_class = None

        # check all classes
        for class_name, rules in classes.items():
            class_markers = set(rules["conv_markers"])
            class_blabels = set(rules["business_labels"])

            # Condition:
            # marker ∈ class markers  AND  business ∈ class business labels
            if turn_markers & class_markers and turn_blabels & class_blabels:
                matched_class = class_name
                break  # stop on first match (or remove break for multi-label mapping)

        results.append({
            "turn_idx": i,
            "turn_text": turn["text"],
            "turn_conv_marker": list(turn_markers),
            "turn_business_label": list(turn_blabels),
            "assigned_class": matched_class
        })

    return results


# how to use fuctions

# span_index = get_span_indices(preprocessed_data, threshold=80, top_k=3)
# print(span_index)
# span_group = group_consecutive(span_index)
# span = get_span_turns(span_group)
# print(span)

# sent_vec = get_sentiment_vector(transcript_dict)
# print(sent_vec)

# mapped = map_turns_to_classes(turns, classes)

# input_csv = "transcript.csv"               # <-- your existing CSV
# output_csv = "transcript.csv"              # <-- overwrite same CSV

# rows = []

# with open(input_csv, "r", encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     transcripts_list = list(reader)

# for row in transcripts_list:

#     tid = row["transcript"]    # transcript ID from CSV

#     data = transcript_dict[tid]  # lookup its data structure

#     # Compute span
#     span_idx = get_span_indices(data["preprocessed_data"], threshold=80, top_k=3)
#     span_group = group_consecutive(span_idx)
#     span = get_span_turns(span_group)

#     # Compute sentiment vector
#     sent_vec = get_sentiment_vector(data["transcript_dict"])

#     # Compute flows
#     mapped = map_turns_to_classes(data["turns"], classes)

#     # Structure to store
#     info = {
#         "span": span,
#         "flows": mapped,
#         "temporal_vector": sent_vec
#     }

#     row["analysis"] = json.dumps(info)

#     rows.append(row)


# # Write CSV
# with open("transcript_analysis.csv" # put transcript path here
#           , "w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=["transcript", "analysis"])  # change column names
#     writer.writeheader()
#     writer.writerows(rows)