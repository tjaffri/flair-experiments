from flair.data import Sentence
from flair.models import SequenceTagger

texts = [
    'Hello, World',
    'Lorem ipsum dolor sit amet'
]

tagger = SequenceTagger.load('./temp/best-model.pt')

for text in texts:
    # predict NER tags
    sentence = Sentence(text)
    tagger.predict(sentence)

    print(f'****** {text}')
    spans = sentence.get_spans('ner')
    if not spans:
        print (f'No entities found')

    for entity in spans:
        print({"start": entity.start_pos, "end": entity.end_pos, "label": entity.tag})

    print('****\n')