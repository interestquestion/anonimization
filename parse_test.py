import spacy
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, Doc

SPACY_MODEL = spacy.load("ru_core_news_lg")

text = "Мама мыла раму и Климова П.И. и климова петра иваныча и климова Л.и. после обеда"

doc = SPACY_MODEL(text)

for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text, token.ent_type_)
    # print positions of words in the text
    print(token.idx, token.idx + len(token.text))

natasha_segmenter = Segmenter()
natasha_morph_vocab = MorphVocab()
natasha_news_embedding = NewsEmbedding()
natasha_morph_tagger = NewsMorphTagger(natasha_news_embedding)
natasha_syntax_parser = NewsSyntaxParser(natasha_news_embedding)
natasha_ner_tagger = NewsNERTagger(natasha_news_embedding)

doc = Doc(text)
doc.segment(natasha_segmenter)
doc.tag_morph(natasha_morph_tagger)
doc.parse_syntax(natasha_syntax_parser)
doc.tag_ner(natasha_ner_tagger)

print(doc.spans)
doc.ner.print()

from pymystem3 import Mystem

m = Mystem()

analyze = m.analyze(text)

first_name = None
second_name = None
middle_name = None

for word in analyze:
    try:
        analysis = word["analysis"][0]
    except KeyError:
        continue

    if "имя" in analysis["gr"]:
        first_name = word["text"].capitalize()
    elif "фам" in analysis["gr"]:
        second_name = word["text"].capitalize()
    elif "отч" in analysis["gr"]:
        middle_name = word["text"].capitalize()

print(f"{second_name} {first_name} {middle_name}")
