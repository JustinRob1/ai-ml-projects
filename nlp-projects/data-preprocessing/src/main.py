import re
from pathlib import Path

def main():
    clean_data()
    transform_data()
    

# Sources: https://www.w3schools.com/python/python_regex.asp
# https://docs.python.org/3/library/re.html
# https://docs.python.org/3/library/pathlib.html
# https://switowski.com/blog/pathlib/

# Recursively reads every .cha file in all sub-directories of Data and 
# writes the cleaned files to the clean/ directory
def clean_data():
    clean_dir = Path('./clean')
    data_dir = Path('./Data')  
    
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Recursively gets the paths of all .cha files in the Data directory
    cha_files = list(data_dir.rglob('*.cha'))
        
    # Regex pattern to match every line that starts with * and followed by 
    # three capital letters which indicates a speaker.
    dialouge_pattern = re.compile(r'\*[A-Z]{3}:\t(.*)')
    
    # Iterates through all the .cha files and reads the lines that match the
    # the dialouge regex. Each line is then cleaned and written to a new file
    for file in cha_files:
        # Source: https://www.reddit.com/r/learnpython/comments/wgzx2w/create_file_if_it_doesnt_exist_as_well_as_its/
        # Creates the relative path of each file and the corresponding directory in the clean directory
        relative_path = file.relative_to(data_dir)
        clean_path = clean_dir / relative_path.with_suffix('.txt')
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file, 'r') as f:
            with open(clean_path, 'a') as clean_file:
                for line in f:
                    match = dialouge_pattern.match(line)
                    if match:
                        # Removes all non-alphanumeric characters, such as punctuation
                        cleaned = re.sub(r"[^\w\s'-]", '', match.group(1))
                        # Removes all digits
                        cleaned = re.sub(r'[\d]', '', cleaned)
                        # Replaces all underscores with spaces
                        cleaned = cleaned.replace('_', ' ')
                        # Removes all extra whitespace
                        cleaned = re.sub(r'\s+', ' ', cleaned)
                        
                        # Writes the cleaned dialouge to the clean file
                        clean_file.write(cleaned + '\n')


# Recursively reads every .txt file in all sub-directories of clean and 
# writes the transfromed files to the transformed/ directory
def transform_data():
    dict = read_dict('cmudict-0.7b')
    not_found = set()

    clean_dir = Path('./clean')
    transformed_dir = Path('./transformed')

    transformed_dir.mkdir(parents=True, exist_ok=True)

    # Recursively gets the paths of all the txt files in the clean directory
    clean_files = list(clean_dir.rglob('*.txt'))

    # Iterates through all the clean files and writes the transformation 
    # to the transformed file
    for file in clean_files:
        # Source: https://www.reddit.com/r/learnpython/comments/wgzx2w/create_file_if_it_doesnt_exist_as_well_as_its/
        # Creates the relative path of each file and the corresponding directory in the transformed directory
        
        relative_path = file.relative_to(clean_dir)
        transformed_path = transformed_dir / relative_path.with_suffix('.txt')
        transformed_path.parent.mkdir(parents=True, exist_ok=True)

        with open(transformed_path, 'a') as transformed_file:
            with open(file, 'r') as f:
                for line in f:
                    transformed = ""
                    for word in line.split():
                        if word.upper() in dict.keys():
                            transformation = dict[word.upper()]
                            transformed += re.sub(r'\d', '', transformation) + " "
                        else:
                            transformation = transform_absent_word(word, dict)
                            transformed += transformation
                    transformed_file.write(transformed + '\n')
    

 # Performs a sequence of checks to match any missing patterns
 # that are absent in the CMU Dictionary        
def transform_absent_word(word, dict):
    # Sequence of checks to match any missing patterns

    # For words that are plural, but not in dictionary
    plural_pattern = re.compile(r"(\w+)s")
    plural_match = plural_pattern.match(word)
    if plural_match:
        try:
            transformation = dict[plural_match.group(1).upper()]
            # substitute any lexical stress markers
            transformed = re.sub(r'\d', '', transformation) + " Z "
        except:
            transformed = "<NF> "
        return transformed


    # For words that have "'s"
    apostrophe_pattern = re.compile(r"(.*)'s")  # commonly (name)'s
    apostrophe_match = apostrophe_pattern.match(word)
    if apostrophe_match:
        try:
            transformation = dict[apostrophe_match.group(1).upper()]
            # substitute any lexical stress markers
            transformed = re.sub(r'\d', '', transformation) + " Z "
        except:
            transformed = "<NF> "
        return transformed
    
    # For words that have one or more hyphens 
    hyphen_pattern = re.compile(r"-?(\w+)-?")
    hyphen_match = hyphen_pattern.findall(word)
    if hyphen_match:
        transformed = ""
        for match_word in hyphen_match:
            try:
                transformation = dict[match_word.upper()]
                # substitute any lexical stress markers
                transformed += re.sub(r'\d', '', transformation) + " "
            except:
                transformed += "<NF> "
        return transformed
                
# Prepares a Python Dictionary for CMU Transformations
def read_dict(path):

    # Source : https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte
    with open(path, "r", encoding="latin-1") as dict_file:
        data = dict_file.read()
    # finds and captures all the CMU Dictionary word and word translation pairs
    dict = re.findall(r"([!\"#%&\(\)\{\}+,\.\:\;\?\/\d_A-Z'-]+)\s+(.*)", data, re.MULTILINE)
    dict = {a:b for a, b in dict}
    return dict

if __name__ == "__main__":
    main()