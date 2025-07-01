import random 
import re, os
import pandas as pd
from typing import List, Dict, Any, Optional
from hf_fewshot.prompting_utils import load_jsonlines, write_jsonlines, load_json, write_json 

def sample_random_pairs(num_elements, num_comps, max_attempts=10000): 
    """
    Randomly generates a set of unique pairs from a given number of elements, ensuring that each element is included in a specified number of comparisons.
    Args:
        num_elements (int): The total number of elements to generate pairs from.
        num_comps (int): The target number of comparisons each element should be included in.
        max_attempts (int, optional): The maximum number of attempts to generate pairs before stopping. Defaults to 10000.
    Returns:
        tuple: A tuple containing:
            - selected_pairs (set): A set of unique pairs (tuples) of element indices.
            - array_count (list): A list where each index represents an element and the value represents the number of times that element has been included in a pair.
    Raises:
        ValueError: If num_elements or num_comps is less than 1.
    Notes:
        - The function will print progress updates every 1000 attempts.
        - If the function cannot reach the target number of comparisons for each element within the maximum number of attempts, it will print a warning message.
    """
    # Initialize counts for how many times each element has been included in pairs
    array_count = [0] * num_elements
    selected_pairs = set()
    
    attempts = 0
    
    # Randomly generate pairs based on index
    while min(array_count) < num_comps and attempts < max_attempts:
        attempts += 1
        elem1 = random.randint(0, num_elements - 1)
        elem2 = random.randint(0, num_elements - 1)
        
        if elem1 != elem2:
            # sort the pair to ensure a uniform order of indices (smaller, larger)
            pair = tuple(sorted((elem1, elem2)))
            
            # If the pair has not been selected and at least one of the elements has not been included in enough pairs
            if (pair not in selected_pairs and
                (array_count[elem1] < num_comps or array_count[elem2] < num_comps)):
                
                selected_pairs.add(pair)
                array_count[elem1] += 1
                array_count[elem2] += 1
                attempts = 0  # Reset attempts counter after successful addition

        # Display progress
        if attempts % 1000 == 0:
            print(f"\r{len(selected_pairs)} pairs selected", end="")
    
    print(f"\nTotal selected pairs: {len(selected_pairs)}")
    if min(array_count) < num_comps:
        print(f"Warning: Could not reach target comparisons. Min comparisons: {min(array_count)}")
    
    return selected_pairs, array_count


def load_data(
    file_path: str,
    file_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load data from CSV or JSONLines file and return a list of dictionaries.
    
    Args:
        file_path: Path to the data file
        file_type: Type of file ('csv' or 'jsonl'). If None, inferred from file extension
            
    Returns:
        List of dictionaries representing the data
        
    Raises:
        ValueError: If file_type is not supported or can't be inferred
        FileNotFoundError: If the file doesn't exist
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Infer file type from extension if not specified
    if file_type is None:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.csv':
            file_type = 'csv'
        elif ext in ('.jsonl', '.json', '.jl'):
            file_type = 'jsonl'
        else:
            raise ValueError(f"Could not infer file type from extension: {ext}")
    
    # Load data based on file type
    if file_type == 'csv':
        # Use pandas to read CSV
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    elif file_type in ('jsonl', 'json', 'jl'):
        # Use huggingface_fewshot to read JSONLines
        return load_jsonlines(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# write a function to create the data 

def create_pairwise_dataset(
        items: list,
        output_file: str,
        num_comps: int,
        id_key: str = "id",
        text_key: str = "text",
): 
    """
    Create a pairwise dataset from a list of items. 
    Args:
        items (list): A list of items to compare. Each item should be a dictionary with an id_key and text_key.
        output_file (str): The output file path for the pairwise dataset.
        num_comps (int): The number of comparisons each item should be included in.
        id_key (str, optional): The key in the item dictionary that represents the item's ID. Defaults to "id".
        text_key (str, optional): The key in the item dictionary that represents the item's text. Defaults to "text".
    """
    
    # TODO: 
    # first, find out if there are repeated elements in the items list
    # if so, create unique items out of them so that they are not repeated in the pairwise comparisons
    # note the frequency of each item 
    
    num_items = len(items)
    

    # Generate pairs
    selected_pairs, array_count = sample_random_pairs(num_items, num_comps)
    
    # Create pairwise dataset
    pairwise_data = []
    for pair in selected_pairs:
        item1 = items[pair[0]]
        item2 = items[pair[1]]
        id1, id2 = str(item1[id_key]), str(item2[id_key])
        
        pairwise_data.append({
            "id": f"{id1}_{id2}",  
            "id1": id1,
            "text1": item1[text_key],
            "id2": id2,
            "text2": item2[text_key],
        })
    
    # Write to file
    write_jsonlines(pairwise_data, output_file)
    print(f"Pairwise dataset saved to: {output_file}")


