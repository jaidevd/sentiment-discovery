import torch
import json
from collections import Counter
import copy
from model.sentiment_classifier import SentimentClassifier
from torch.autograd import Variable
from data_utils.loaders import DataLoader
import data_utils
from tqdm import tqdm
import pandas as pd
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
import numpy as np
from joblib import load
from sqlalchemy import create_engine


CUDA = False
REMOVE = {'https'} | ENGLISH_STOP_WORDS
nlp = spacy.load('en_core_web_sm')


def load_model(mpath):
    sd = torch.load(mpath, map_location='cpu')
    if 'args' in sd:
        model_args = sd['args']
    if 'sd' in sd:
        sd = sd['sd']
    ntokens = model_args.data_size
    concat_pools = model_args.concat_max, model_args.concat_min, \
        model_args.concat_mean

    model = SentimentClassifier(model_args.model, ntokens, model_args.emsize,
                                model_args.nhid, model_args.nlayers,
                                model_args.classifier_hidden_layers,
                                model_args.classifier_dropout,
                                model_args.all_layers,
                                concat_pools, False, model_args)
    if CUDA:
        model.cuda()  # -- see how well this works
    model.load_state_dict(sd)
    model.set_neurons(1)
    return model


def load_data(fpath, tcol="text"):
    """makes training/val/test"""
    batch_size = 128
    seq_length = 256
    eval_seq_length = 256
    data_loader_args = {'num_workers': 4, 'shuffle': False,
                        'batch_size': batch_size,
                        'pin_memory': True, 'transpose': False,
                        'distributed': False,
                        'rank': -1, 'world_size': 1,
                        'drop_last': False}
    split = [1.0, 0.0, 0.0]
    data_set_args = {
        'path': [fpath], 'seq_length': seq_length, 'lazy': False, 'delim': ',',
        'text_key': tcol, 'label_key': 'label', 'preprocess': False,
        'ds_type': 'supervised', 'split': split, 'loose': False,
        'tokenizer_type': 'CharacterLevelTokenizer',
        'tokenizer_model_path': 'tokenizer.model',
        'vocab_size': 256, 'model_type': 'bpe',
        'non_binary_cols': None, 'process_fn': 'process_str'}

    # eval_loader_args = copy.copy(data_loader_args)
    eval_set_args = copy.copy(data_set_args)
    eval_set_args['split'] = [1.]

    eval_set_args['seq_length'] = eval_seq_length

    train = None
    valid = None
    test = None

    train, tokenizer = data_utils.make_dataset(**data_set_args)
    eval_set_args['tokenizer'] = tokenizer

    if train is not None:
        train = DataLoader(train, **data_loader_args)
    return (train, valid, test), tokenizer


def classify(model, text):
    # Make sure to set *both* parts of the model to .eval() mode.
    model.lm_encoder.eval()
    model.classifier.eval()
    # Initialize data, append results
    stds = np.array([])
    labels = np.array([])
    label_probs = np.array([])
    first_label = True
    heads_per_class = 1

    def get_batch(batch):
        text = batch['text'][0]
        timesteps = batch['length']
        labels = batch['label']
        text = Variable(text).long()
        timesteps = Variable(timesteps).long()
        labels = Variable(labels).long()
        if CUDA:
            text, timesteps, labels = text.cuda(), timesteps.cuda(), \
                labels.cuda()
        return text.t(), labels, timesteps - 1

    def get_outs(text_batch, length_batch):
        model.lm_encoder.rnn.reset_hidden(128)
        return model(text_batch, length_batch, False)

    with torch.no_grad():
        for i, data in tqdm(enumerate(text), total=len(text)):
            text_batch, labels_batch, length_batch = get_batch(data)
            # get predicted probabilities given transposed text and
            # lengths of text
            probs, _ = get_outs(text_batch, length_batch)
#            probs = model(text_batch, length_batch)
            if first_label:
                first_label = False
                labels = []
                label_probs = []
                if heads_per_class > 1:
                    stds = []
            # Save variances, and predictions
            # TODO: Handle multi-head [multiple classes out]
            if heads_per_class > 1:
                _, probs, std, preds = probs
                stds.append(std.data.cpu().numpy())
            else:
                probs, preds = probs
            labels.append(preds.data.cpu().numpy())
            label_probs.append(probs.data.cpu().numpy())

    if not first_label:
        labels = (np.concatenate(labels))  # .flatten())
        label_probs = (np.concatenate(label_probs))  # .flatten())
        if heads_per_class > 1:
            stds = (np.concatenate(stds))
        else:
            stds = np.zeros_like(labels)
    return labels, label_probs, stds


sentiment = {
    'anger': 'Negative',
    'anticipation': 'Neutral',  # is this contributing to so many Positive?
    'disgust': 'Negative',
    'fear': 'Negative',
    'joy': 'Positive',
    'sadness': 'Negative',
    'surprise': 'Neutral',
    'trust': 'Neutral'
}
cols = list(sentiment.keys())


def labelfinder(row):
    mask = row[cols] > 0.5
    if mask.any():
        if mask.sum() > 1:
            fines = mask.index[mask.values]
            # check if the multiple fine sentiments map to the same coarse one.
            coarse = set()
            for sent in fines:
                coarse.add(sentiment[sent])
            if len(coarse) == 1:
                label = coarse.pop()
                prob = row[fines].mean()
            else:
                fine = row[fines].astype(float).idxmax()
                label = sentiment[fine]
                prob = row[fine]
        else:
            fine = mask.index[mask.values][0]
            label = sentiment[fine]
            prob = row[fine]
    else:
        label = 'Neutral'
        prob = (random.random() + 1) / 2
    return label, prob


def sentiment_discovery(fpath, tcol="text"):
    model = load_model('mlstm_semeval.clf')
    data = load_data(fpath, tcol)
    (train, val, test), tokenizer = data
    ypred, yprob, ystd = classify(model, train)
    sdf = pd.DataFrame(yprob, columns=cols)
    sdf['text'] = pd.read_csv(fpath, usecols=['text'], squeeze=True)
    labelprob = sdf.apply(labelfinder, axis=1)
    sdf['label'] = labelprob.apply(lambda x: x[0])
    sdf['prob'] = labelprob.apply(lambda x: x[1])
    return sdf


def vader(df, an=None, tcol='text'):
    if not an:
        an = SentimentIntensityAnalyzer()
    recs = []
    for ix, rowdata in df.iterrows():
        p = an.polarity_scores(rowdata[tcol])
        p['Positive'] = p.pop('pos')
        p['Negative'] = p.pop('neg')
        p['Neutral'] = p.pop('neu')
        recs.append(p)
    recs = []
    scores = pd.io.json.json_normalize(recs)
    df = pd.concat([df, scores], axis=1)
    df['label'] = df[['Neutral', 'Positive', 'Negative']].idxmax(axis=1)
    df['score'] = df[['Neutral', 'Positive', 'Negative']].max(axis=1)
    return df


def make_sentiment_pie(df, col="label", out="sentiment_pie.json"):
    xx = pd.DataFrame(df[col].value_counts()).reset_index()
    xx.columns = ['label', 'metric']
    xx.to_json(out, orient="records")


def make_positive_wc(df, tcol="text", lcol="label", out='pos_wordcloud.json'):
    xdf = df[df[lcol] == 'Positive']
    sents = xdf['text'].tolist()
    words = []
    for s in sents:
        doc = nlp(s)
        words.extend([c.text for c in doc if tokenmatch(c)])

    counter = Counter(words).most_common(200)
    xx = pd.DataFrame(counter)
    xx.columns = ['Word', 'Frequency']
    xx['Positive_Score'] = make_wordscore(xx, 'pos')
    xx.to_json(out, orient="records")


def make_negative_wc(df, tcol="text", lcol="label", out='neg_wordcloud.json'):
    xdf = df[df[lcol] == 'Negative']
    sents = xdf['text'].tolist()
    words = []
    for s in sents:
        doc = nlp(s)
        words.extend([c.text for c in doc if tokenmatch(c)])

    counter = Counter(words).most_common(200)
    xx = pd.DataFrame(counter)
    xx.columns = ['Word', 'Frequency']
    xx['Negative_Score'] = make_wordscore(xx, 'neg')
    xx.to_json(out, orient="records")


def tokenmatch(c):
    return (c.pos_ in ('NOUN', 'PROPN')) and (not c.is_stop) and \
        (not c.is_punct) and c.is_ascii and len(c.text) > 2


def make_top_tweeters(df, out="tweetconfig.json"):
    payload = []

    # all tweets
    total = {'id': '#tweeters', 'total': df.shape[0], 'col1_header': 'Handle',
             'col2_header': 'Count'}
    total['data'] = []
    for label, value in df['user_name'].value_counts().head().to_dict().items():  # NOQA: E501
        total['data'].append({"label": '@' + label, "value": value})
    payload.append(total)

    # pos tweets
    xdf = df[df['label'] == 'Positive']
    pos = {'id': '#pos_tweeters', 'total': xdf.shape[0],
           'col1_header': 'Handle',
           'col2_header': 'Count'}
    pos['data'] = []
    for label, value in xdf['user_name'].value_counts().head().to_dict().items():  # NOQA: E501
        pos['data'].append({"label": '@' + label, "value": value})
    payload.append(pos)

    # neg tweets
    xdf = df[df['label'] == 'Negative']
    neg = {'id': '#neg_tweeters', 'total': xdf.shape[0],
           'col1_header': 'Handle',
           'col2_header': 'Count'}
    neg['data'] = []
    for label, value in xdf['user_name'].value_counts().head().to_dict().items():  # NOQA: E501
        neg['data'].append({"label": '@' + label, "value": value})
    payload.append(neg)

    with open(out, 'w') as fout:
        json.dump(payload, fout, indent=4)


def make_wordscore(df, sentiment):
    x = np.random.rand(df.shape[0])
    if sentiment == 'neg':
        x *= -1
    return x


def get_indian_tweets(df, clfpath="india-pk.pkl"):
    clf = load(clfpath)
    df['pred'] = clf.predict(df['text'].tolist())
    df = df[~df['pred'].astype(bool)]
    spells = 'pakistan pkistan'.split()
    for s in spells:
        df.drop(
            df.index[df['user_location'].str.contains(
                s, case=False).values.astype(bool)],
            inplace=True, axis=0)
    return df


def make_category_pie(df, lenc_path='topic-lenc.pkl',
                      pipe_path='topic-model.pkl',
                      out='category_sentiment_pie.json'):
    lenc = load(lenc_path)
    pipe = load(pipe_path)
    topics = pipe.predict(df['text'].tolist())
    df['predicted_cat'] = lenc.inverse_transform(topics).tolist()
    xx = pd.DataFrame(df['predicted_cat'].value_counts()).reset_index()
    xx.columns = ['label', 'metric']
    xx.to_json(out, orient="records")


def get_pos_sent_prob(df):
    return df.apply(
        lambda x: 1 - x['prob'] if x['label'] != 'Positive' else x['prob'],
        axis=1).mean()


def make_sentiment_over_time(df, timecol="created_at",
                             out="avg_sentiment_trend.json",
                             mode="hourly"):
    df[timecol] = pd.to_datetime(df[timecol], utc=True)
    latest = df[timecol].max()
    payload = []
    if mode == "hourly":
        for hstart in [5, 6, 7, 8, 9]:
            start = pd.Timestamp(year=latest.year, month=latest.month,
                                 day=latest.day, hour=hstart, minute=0,
                                 tz='UTC')
            end = pd.Timestamp(year=latest.year, month=latest.month,
                               day=latest.day, hour=hstart + 1, minute=0,
                               tz='UTC')
            tslice = df[df[timecol] < end]
            tslice = tslice[tslice[timecol] >= start]
            xaxis_band = end.tz_convert('Asia/Kolkata').strftime("%I:%M %p")
            for cat, count in tslice['predicted_cat'].value_counts().items():
                xdf = tslice[tslice['predicted_cat'] == cat]
                score = get_pos_sent_prob(xdf[['label', 'prob']])
                payload.append({'category': cat, 'tweets_count': count,
                                'avg_sentiment_score': score,
                                'xaxis_bands': xaxis_band})
    elif mode == "daily":
        for i in range(6, 0, -1):
            day = latest - pd.Timedelta(i, unit="D")
            start = pd.Timestamp(
                year=day.year, month=day.month, day=day.day, hour=0, tz='UTC')
            end = pd.Timestamp(
                year=day.year, month=day.month, day=day.day, hour=23,
                minute=59, second=59, tz='UTC')
            tslice = df[df[timecol] < end]
            tslice = tslice[tslice[timecol] >= start]
            xaxis_band = end.tz_convert('Asia/Kolkata').strftime("%d %b")
            for cat, count in tslice['predicted_cat'].value_counts().items():
                xdf = tslice[tslice['predicted_cat'] == cat]
                score = get_pos_sent_prob(xdf[['label', 'prob']])
                payload.append({'category': cat, 'tweets_count': count,
                                'avg_sentiment_score': score,
                                'xaxis_bands': xaxis_band})
    with open(out, 'w') as fout:
        json.dump(payload, fout, indent=4)


def make_sentiment_count(df, timecol="created_at",
                         out="count_sentiment_stepchart.json",
                         mode="hourly"):
    df[timecol] = pd.to_datetime(df[timecol], utc=True)
    latest = df[timecol].max()
    payload = []
    if mode == "hourly":
        for hstart in [5, 6, 7, 8, 9]:
            start = pd.Timestamp(year=latest.year, month=latest.month,
                                 day=latest.day, hour=hstart, minute=0,
                                 tz='UTC')
            end = pd.Timestamp(year=latest.year, month=latest.month,
                               day=latest.day, hour=hstart + 1, minute=0,
                               tz='UTC')
            tslice = df[df[timecol] < end]
            tslice = tslice[tslice[timecol] >= start]
            xaxis_band = end.tz_convert('Asia/Kolkata').strftime("%I:%M %p")
            for cat, count in tslice['predicted_cat'].value_counts().items():
                xdf = tslice[tslice['predicted_cat'] == cat]
                scounts = xdf['label'].value_counts().to_dict()
                scounts = {k + ' Sentiment Count': v for k, v in scounts.items()}  # NOQA: E501
                scounts['category'] = cat
                scounts['xaxis_bands'] = xaxis_band
                payload.append(scounts)
    elif mode == "daily":
        for i in range(6, 0, -1):
            day = latest - pd.Timedelta(i, unit="D")
            start = pd.Timestamp(
                year=day.year, month=day.month, day=day.day, hour=0, tz='UTC')
            end = pd.Timestamp(
                year=day.year, month=day.month, day=day.day, hour=23,
                minute=59, second=59, tz='UTC')
            tslice = df[df[timecol] < end]
            tslice = tslice[tslice[timecol] >= start]
            xaxis_band = end.tz_convert('Asia/Kolkata').strftime("%d %b")
            for cat, count in tslice['predicted_cat'].value_counts().items():
                xdf = tslice[tslice['predicted_cat'] == cat]
                scounts = xdf['label'].value_counts().to_dict()
                scounts = {k + ' Sentiment Count': v for k, v in scounts.items()}  # NOQA: E501
                scounts['category'] = cat
                scounts['xaxis_bands'] = xaxis_band
                payload.append(scounts)
    for p in payload:
        for s in 'Positive Negative Neutral'.split():
            key = s + ' Sentiment Count'
            if key not in p:
                p[key] = 0
    with open(out, 'w') as fout:
        json.dump(payload, fout, indent=4)


def main(fpath):
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
    elif fpath.endswith('.db'):
        engine = create_engine(fpath)
        df = pd.read_sql_table('Tweets', engine)
    df.dropna(subset=['text'], inplace=True)
    df = get_indian_tweets(df)
    df.to_csv('/tmp/temp.csv', index=False)
    sdf = sentiment_discovery('/tmp/temp.csv')
    del sdf['text']
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, sdf], axis=1, verify_integrity=True)
    df.to_csv('/tmp/sentiment_cache.csv', index=False)

    make_top_tweeters(df)
    make_sentiment_pie(df)
    make_category_pie(df)

    make_sentiment_over_time(df, mode="hourly")
    make_sentiment_count(df, mode="hourly")

    make_positive_wc(df)
    make_negative_wc(df)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
