import choix
from collections import defaultdict
import re
from hf_fewshot.prompting_utils import load_jsonlines, load_yaml

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

    return [text.lower(), "".join(text.lower().split())] # return a list of variations to match against


def get_pairwise_outcomes(
    cleaned_results: list, 
    index_map: dict,
    response_parser: callable = default_response_parser,
    expand_options: callable = default_expand_options,
    tie_logic: str = "skip", # choices = add_both, skip  
    outcome_map = {
        1: "Text 1", 
        2: "Text 2",
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
        response = result.get('response', None)

        # Use the response_parser if provided
        if response_parser:
            response = response_parser(response)

        # check for different response formats: 
        for key, val in outcome_map: 
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
            dataset_indexed[index_map[id1]].update({'id': id1, 'text': text1})
        if id2 not in dataset_indexed:
            dataset_indexed[index_map[id2]].update({'id': id2, 'text': text2}) 

    nitems = find_nitems(pairwise_outcomes)

    scaled_results = choix.ilsr_pairwise(nitems, pairwise_outcomes, alpha=0.01)

    for index, elem in enumerate(scaled_results):
        dataset_indexed[index]['score'] = elem

    return dataset_indexed





if __name__ == "__main__":
    pairwise_config = load_yaml("../example_pairwise_task/sample_config.yaml")
    
