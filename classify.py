from model.data_utils import Dataset
from model.models import HANNModel
from model.config import Config
import argparse
import os
import sys


def main():

    dataset = "nicta"
    #dataset = "pubmed-20k"
    #dataset = "pubmed-200k"

    model_weights = "results/test/model.weights"

    abstract = '''       
        The sinoatrial ( SA ) and atrioventricular ( AV ) nodes are specialized centers of the heart conduction system and are composed of muscle cells with distinctive morphological and electrophysiological properties .
        We report here results of immunofluorescence and immunoperoxidase studies on the bovine heart showing that a large number of SA and AV nodal cells share a distinct type of myosin heavy chain ( MHC ) which is not found in other myocardial cells and can thus.        
        '''

    abstract_sentences = split_to_sentences(abstract)
    predicted_labels = predict(model_weights, abstract_sentences)

    print()
    print("predicted classes")
    for (label, sentence) in zip(predicted_labels, abstract_sentences):
        print(label.upper() + ": " + sentence) 

def split_to_sentences(text):
    sentences = []
    for l in text.split("."):
        l = l.strip() + "."
        if len(l) > 1:
            sentences += [l]
    return sentences 

def predict(weights_path, abstract_sentences):

    sys.argv.extend(['--dataset_name', "nicta"])
    parser = argparse.ArgumentParser()

    config = Config(parser)

    # restore model weights
    model = HANNModel(config)
    model.build()
    model.restore_session("results/test/model.weights") 


    sentences_words = []
    # split abstract to sentences
    for line in abstract_sentences:
        # split line into words and map  words to ids
        sentence = [config.processing_word(word) for word in line.split()]
        sentences_words += [sentence]

    # run prediction
    labels_pred, _ = model.predict_batch([sentences_words])

    # map: label id to label string
    tag_id_to_label = dict((v,k) for k,v in config.vocab_tags.items())
    
    # convert predicted labels to string
    labels_pred_str = []
    for sublist in labels_pred:
        for item in sublist:
            labels_pred_str.append(tag_id_to_label[item])
    
    return labels_pred_str



if __name__ == "__main__":
    main()
