"""
File for preprocessing and augmenting data
"""
from bleach import clean
import pandas as pd
import argparse
import preprocessor as p  # forming a separate feature for cleaned tweets
from nlpaug.augmenter.word.synonym import SynonymAug
from nlpaug.augmenter.word.back_translation import BackTranslationAug

parser = argparse.ArgumentParser(description='Arguments for preprocessing the data.')
parser.add_argument('-data_path', type=str, default='../datasets/tweet_emotions.csv',
                    help='path to where the data is stored.')
parser.add_argument('-augmentation', type=int, default=0,
                    help='Whether to augment the data or not.')
parser.add_argument('-last_k', type=int, default=4,
                    help='Which least populated columns to augment.')
parser.add_argument('-augmenter', type=str, default='synonym',
                    help='Which augmenter to use.')

def clean_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the tweets
    """
    df.drop(['tweet_id'],axis=1,inplace=True)
    df['content'] = df.content.apply(lambda x: p.clean(x))
    return df

def augment(df:pd.DataFrame,last_k:int,augmenter='synonym')->pd.DataFrame:
    """
    Function for word lvel synonym augmenting string data
    in a DataFrame
    """
    #create the augmenter
    if augmenter=='synonym':
        augmenter = SynonymAug(aug_p=0.2,aug_min=1,aug_max=4)
    else:
        #instantiate the backwards translation
        augmenter = BackTranslationAug()

    #loop over columns and add their augmented versions
    for value in df.sentiment.value_counts().index.to_list()[-last_k:]:
        df_part=df[df['sentiment']==value].copy()
        df_part.content.apply(lambda x: augmenter.augment(x,num_thread=4))
        df=pd.concat([df,df_part])

    return df
# TODO add NLP augmentation
# TODO evaluate model and choose which features to keep
# TODO ADD requirements at the end



if __name__ == '__main__':
    args = parser.parse_args()
    df=pd.read_csv(args.data_path)
    df=clean_tweets(df)
    if args.augmentation:
        df=augment(df,args.last_k,augmenter=args.augmenter)
        df.to_csv('../datasets/preprocessed.csv',index=False)

