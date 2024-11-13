import torch
from maverick import Maverick

# Check for GPU
def check_gpu():
    if torch.cuda.is_available():
        print('Using GPU')
        device = 'cuda:0'
    else:
        print('Using CPU')
        device = 'cpu'

    return device


def test(sentence):
    '''Perform coreference resolution and return the resolved sentence.'''
    device = check_gpu()
    model = Maverick(hf_name_or_path="sapienzanlp/maverick-mes-ontonotes", device=device) 

    result = model.predict(sentence)
    print(f'Coreference resolution result: {result}')

    # Use 'clusters_token_text' as it's available in your result
    resolved_text = sentence
    clusters = result.get('clusters_token_text', [])  # Use an empty list if the key doesn't exist
    for cluster in clusters:
        main_mention = cluster[0]  # The main mention
        for mention in cluster[1:]:  # Replace other mentions with the main mention
            resolved_text = resolved_text.replace(mention, main_mention)

    return resolved_text

    

if __name__ == '__main__':
    print('cuda version:', torch.version.cuda)

    output_file = 'coreference_test_output.txt'
    
    with open(output_file, 'w') as f:


        sentence = "My sister has a dog. She loves him."
        f.write(f'Input sentence: {sentence}\n')
        f.write(f'Output sentence: {test(sentence)}\n')

        sentence = "When Tom met Anna, he offered her a ride home because she missed the bus."
        f.write(f'Input sentence: {sentence}\n')
        f.write(f'Output sentence: {test(sentence)}\n')

        sentence = "The dog chased its tail until it got tired."
        f.write(f'Input sentence: {sentence}\n')
        f.write(f'Output sentence: {test(sentence)}\n')

        paragraph = "Jessica brought her new puppy, Max, to the park. He was full of energy and chased every ball she threw. After a while, a few children came over and wanted to play with him. Jessica watched as they ran around, laughing and cheering every time he fetched the ball. She was glad to see Max enjoying himself and felt proud of how well he behaved. Eventually, she decided it was time to go, so she called him, and he ran back to her with his tail wagging."
        f.write(f'Input paragraph: {paragraph}\n')
        f.write(f'Output paragraph: {test(paragraph)}\n')

