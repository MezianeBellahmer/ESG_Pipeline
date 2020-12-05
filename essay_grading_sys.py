import sys
from src import *
import language_check


def main():
    file = open(sys.argv[1], 'r')
    text = file.read()
    file.close()
   

    file = open(sys.argv[2], 'r')
    prompt = file.read()
    file.close()
  

    df = pd.DataFrame()
    df['essay'] = [text]
    df['Prompt'] = [prompt]

    tool = language_check.LanguageTool('en-US')
    df['matches'] = df['essay'].apply(lambda txt: tool.check(txt))
    df['corrections'] = df.apply(lambda l: len(l['matches']), axis=1)
    df['corrected'] = df.apply(lambda l: language_check.correct(l['essay'], l['matches']), axis=1)
    df['word_count'] = words_count(df, 'essay')

    # extract average sentence length, sentence's count, and sentences
    df['avrg_sents_length'] = avrg_sents_length(df, 'corrected')
    df['sents_count'] = sents_count(df, 'corrected')
    df['sents'] = sentences(df, 'corrected')

    df['topic_detection'], df['lexical_divr'], df['fk_score'], df['prompt_sim'] = some_func(df)

    # counting the punctuation
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    df['count_punct'] = df.essay.apply(lambda s: count(s, string.punctuation))

    df['Polarity'] = polarity(df, 'corrected')
    df['Subjectivity'] = subjectivity(df, 'corrected')

    df['Avg_tree_height'] = avg_tree_height(df, 'corrected')
    
    df['inner_similarities'] = essay_similarity(df, 'sents')

    df = text_coherence_DF(df, text_column_name='essay')
    
    print(df)


if __name__ == "__main__":
    main()