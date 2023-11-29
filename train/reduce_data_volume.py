import common
import json

def remove_posts_starting_with_at(data):
    # Remove every post where the final message content starts with @
    # Iterate through the training data
    y = []
    for i in range(len(data)):
        # Get the final message
        final_message = data[i]['messages'][-1]['content']

        # If the final message starts with @
        if final_message[0] != '@':
            # add it to the new training data
            y.append(data[i])
    
    return y

def write_jsonl(data_list: list, filename: str) -> None:
    """Write a list of dictionaries to a jsonl file, which is the format required by OpenAI."""
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

# Open the training file
for (infn, outfn) in [(common.TRAINING_FILE_NAME, common.TRAINING_NOREPLIES_FILE_NAME),
                       (common.VALIDATION_FILE_NAME, common.VALIDATION_NOREPLIES_FILE_NAME)]:
    with open(infn, 'r') as f:
        jlines = f.read()
        data = [json.loads(jline) for jline in jlines.splitlines()]
        reduced_training_data = remove_posts_starting_with_at(data)
        write_jsonl(reduced_training_data, outfn)