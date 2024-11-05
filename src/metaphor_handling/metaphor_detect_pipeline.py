# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("token-classification", model="lwachowiak/Metaphor-Detection-XLMR")


with open("input_mh.txt", "r") as infile, open("output_md.txt", "w") as outfile:
    for line in infile:
        sentence = line.strip()
        

        result = pipe(sentence)
        label_list = []
        sentence_label = 0
        for dict in result:
            #print(dict['entity'])
            if dict['entity'] == 'LABEL_0':
                label_list.append(0)
            elif dict['entity'] == 'LABEL_1':
                label_list.append(1)
                sentence_label = 1

        print(sentence)
        print(f'{label_list} overall label: {sentence_label}')
        # Write the labeled sentence to the output file
        outfile.write(f"{sentence_label}\t{sentence}\n")
