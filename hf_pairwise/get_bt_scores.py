import choix
from collections import defaultdict
import re
from hf_fewshot.prompting_utils import load_jsonlines, load_yaml, write_jsonlines
from hf_fewshot.classifiers import few_shot_classifier
from hf_pairwise.create_pairwise_dataset import create_pairwise_dataset
import argparse
from pathlib import Path

# first, get the pairwise outcomes and clean them up 
def clean_pairwise_results(
    pairwise_results: list, 
    pairwise_data: list, 
    min_comparisons: int = 5, 
    
):
    """
    Clean and filter pairwise comparison results.
    
    Parameters:
    - pairwise_results: List of dictionaries with comparison results (including ID field with format id1_id2)
    - pairwise_data: List of dictionaries with original comparison data (including id1, id2, text1, text2)
    - min_comparisons: Minimum number of comparisons required for an item
    
    Returns:
    - Filtered list of pairwise results
    - Mapping from original IDs to indices
    """

    # Create a dictionary to map pairs to their comparison data
    pair_to_data = {}
    for item in pairwise_data:
        id1, id2 = str(item['id1']), str(item['id2'])
        key = f"{id1}_{id2}"
        pair_to_data[key] = item

    # Process results to extract id1 and id2 
    id_counts = defaultdict(int)
    processed_results = []
    for result in pairwise_results:
        try:
            id1, id2 = result['id'].split('_')
            # technically, the order of id1 and id2 should not matter
            # but the variations should be in the data 
            result_with_ids = {**result, 'id1': id1, 'id2': id2}
            processed_results.append(result_with_ids)
            # Increment the count for both IDs
            id_counts[id1] += 1
            id_counts[id2] += 1
        
        except ValueError:
            print(f"Warning: Could not parse ID {result['ID']}, skipping this result")
    
    # Filter out pairs that do not meet the minimum comparison threshold
    valid_ids = {id_val for id_val, count in id_counts.items() if count >= min_comparisons}

    # Filter pairwise results to only include valid IDs
    cleaned_results = []
    for result in processed_results:
        id1, id2 = result['id1'], result['id2']
        if id1 in valid_ids and id2 in valid_ids:
            key = f"{id1}_{id2}"
            if key in pair_to_data: 
                # combine the result with the data 
                combined_result = {**result, **pair_to_data[key]}
                cleaned_results.append(combined_result)

    # create a mapping for all valid IDs 
    all_valid_ids = sorted(list(valid_ids))
    index_map = {original_id: idx for idx, original_id in enumerate(all_valid_ids)}

    # Return the cleaned results and the index mapping
    return cleaned_results, index_map 



def default_response_parser(response: str):
    """
    Input: 

    EXPLANATION: adsasd asdffsda dfas
    ANSWER: Text 1 

    Output: Text 1 
    
    ====================================================================
    
    Default parser for response strings to extract the actual outcome.
    
    Parameters:
    - response: The response string from the comparison result. Can contain an explanation or other additional text. 
    
    Returns:
    - The parsed outcome (e.g., "Text 1", "Text 2", etc.)
    
    """
    if not response:
        return None
    
    # Use regex to find the answer part
    match = re.search(r'ANSWER:\s*(\w+)', response, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # If no match found, return None
    print(f"Warning: Could not parse response: {response}")
    return None

def default_expand_options(text : str): 
    """
    The logic to create variations of the text to account for small differences in generated text
    Parameters: 
        - text: The input text to match against

    Returns:
        - A list of variations of the input text to match against 

    Input: "Text 1 "
    Output: text1, text 1

    This function can be made more complex 
    """

    # lowercase and strip whitespace 

    return list(set([text.lower(), "".join(text.lower().split())])) # return a list of variations to match against


def get_pairwise_outcomes(
    cleaned_results: list, 
    index_map: dict,
    response_parser: callable = None, # Optional: a callable to parse the response field
    expand_options: callable = default_expand_options,
    tie_logic: str = "skip", # choices = add_both, skip  
    outcome_map = {
        1: "Q1", 
        2: "Q2",
        3: "Tie",  # Optional: if using a tie-breaker or a neutral option, this can be used to indicate a tie
        4: "NONE" # Optional: The comparison is not valid
    }, 
): 
    """
    Convert cleaned results to pairwise outcomes format for choix.
    
    Parameters:
    - cleaned_results: List of filtered comparison results
    - index_map: Mapping from original IDs to indices
    - response_parser: Optional callable to parse the response field
    
    Returns:
    - List of (winner_idx, loser_idx) tuples
    """
    pairwise_outcomes = [] 

    for result in cleaned_results: 
        id1, id2 = result['id1'], result['id2']

        # skip if neither id1 nor id2 are in the index map
        if id1 not in index_map or id2 not in index_map:
            print(f"Warning: ID pair ({id1}, {id2}) not in index map, skipping")
            continue

        idx1, idx2 = index_map[id1], index_map[id2]

        # Determine the winner and loser based on the outcome
        response = result.get('response', None).lower()

        # Use the response_parser if provided
        if response_parser:
            response = response_parser(response)

        # check for different response formats: 
        for key, val in outcome_map.items(): 
            expanded_options = expand_options(val) # get variations of the text to match against

            if any([x in response for x in expanded_options]) : 
                # if the response matches one of the options, assign the winner and loser
                if key == 1: 
                    pairwise_outcomes.append((idx1, idx2))
                elif key == 2:
                    pairwise_outcomes.append((idx2, idx1))
                elif key == 3: 
                    # tie logic 
                    if tie_logic == "add_both": 
                        pairwise_outcomes.append((idx1, idx2)) # add both outcomes
                        pairwise_outcomes.append((idx2, idx1)) # add the reverse as well
                    elif tie_logic == "skip":
                        continue # skip this outcome, do not add to pairwise outcomes
                elif key == 4: 
                    continue # do not add this outcome, invalid 

                break 

    return pairwise_outcomes 

def find_nitems(pairwise_outcomes: list):
    """
    Find the number of unique items in the pairwise outcomes.
    
    Parameters:
    - pairwise_outcomes: List of (winner_idx, loser_idx) tuples
    
    Returns:
    - Number of unique items
    """
    unique_indices = set()
    for winner, loser in pairwise_outcomes:
        unique_indices.add(winner)
        unique_indices.add(loser)
    
    return len(unique_indices)

def get_bt_scores(pairwise_data, pairwise_results, threshold=2):
    # make sure all ids are strings in the data 
    for elem in pairwise_data: 
        elem['id1'] = str(elem['id1'])
        elem['id2'] = str(elem['id2'])

    cleaned_pairwise_results, index_map = clean_pairwise_results(pairwise_results, pairwise_data, threshold)
    pairwise_outcomes = get_pairwise_outcomes(cleaned_pairwise_results, index_map)
    dataset_indexed = {}

    # Here, we need a mapping of indices to the original data to store the scores
    for elem in cleaned_pairwise_results:
        id1 = str(elem['id1'])
        id2 = str(elem['id2'])

        text1, text2 = elem['text1'], elem['text2']

        if id1 not in dataset_indexed:
            dataset_indexed[index_map[id1]] = {'id': id1, 'text': text1}
        if id2 not in dataset_indexed:
            dataset_indexed[index_map[id2]] = {'id': id2, 'text': text2}

    nitems = find_nitems(pairwise_outcomes)

    scaled_results = choix.ilsr_pairwise(nitems, pairwise_outcomes, alpha=0.01)
    
    for index, elem in enumerate(scaled_results):
        dataset_indexed[index]['score'] = elem

    return dataset_indexed

def add_args(parser):
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the configuration file containing dataset and model details."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get pairwise outcomes and scores")
    add_args(parser)
    args = parser.parse_args()
    config = load_yaml(args.config)
    # first, load the data and create the pairwise dataset 
    dataset= load_jsonlines(config['dataset']['pointwise_data'])
    output_dir = Path(config['output']['output_dir'])
    
    # created from output_dir, one can also specify the output file name in the config
    pairwise_dataset_path = f"{output_dir}/pairwise_dataset.jsonl"

    create_pairwise_dataset(
        items=dataset, 
        output_file=pairwise_dataset_path, 
        num_comps=config['dataset']['min_comparisons'], 
        id_key=config['dataset']['id_key'], 
        text_key=config['dataset']['text_key']
    )

    # load the pairwise dataset
    pairwise_dataset = load_jsonlines(pairwise_dataset_path) 

    config['dataset']['path'] = pairwise_dataset_path

    print(f"Pairwise data loaded from: {config['dataset']['path']}")
    print("Size of the pairwise dataset:", len(pairwise_dataset))
    
    # get the pairwise outcomes 
    few_shot_classifier(config) 


    pairwise_results = load_jsonlines(output_dir / config['output']['output_file'])

    print(f"Loaded {len(pairwise_results)} pairwise outcomes.")

    bt_scores = get_bt_scores(pairwise_dataset, pairwise_results, threshold=2)
    bt_scores_output_path = output_dir / config['output']['output_bt_file']

    # write the bt scores to a jsonlines file
    bt_scores_list = list(bt_scores.values())

    # reverse sort the list by score
    bt_scores_list.sort(key=lambda x: x['score'], reverse=True)

    # save the scores to a jsonlines file
    write_jsonlines(bt_scores_list, bt_scores_output_path)
    print(f"BT scores saved to: {bt_scores_output_path}")
    






    




