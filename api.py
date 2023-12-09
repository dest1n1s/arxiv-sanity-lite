import os
import re
import time
from random import shuffle

import numpy as np
from sklearn import svm

from aslite.db import get_papers_db, get_metas_db, get_tags_db, get_last_active_db, get_email_db
from aslite.db import load_features

pdb = None
mdb = None
RET_NUM = 25

def get_papers():
    if not pdb:
        pdb = get_papers_db()
    return pdb

def get_tags():
    return {}

def get_metas():
    if not mdb:
        mdb = get_metas_db()
    return mdb

def search_rank(q: str = ''):
    if not q:
        return [], [] # no query? no results
    qs = q.lower().strip().split() # split query by spaces and lowercase

    pdb = get_papers()
    match = lambda s: sum(min(3, s.lower().count(qp)) for qp in qs)
    matchu = lambda s: sum(int(s.lower().count(qp) > 0) for qp in qs)
    pairs = []
    for pid, p in pdb.items():
        score = 0.0
        score += 10.0 * matchu(' '.join([a['name'] for a in p['authors']]))
        score += 20.0 * matchu(p['title'])
        score += 1.0 * match(p['summary'])
        if score > 0:
            pairs.append((score, pid))

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return pids, scores

def svm_rank(tags: str = '', pid: str = '', C: float = 0.01):
    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    if not (tags or pid):
        return [], [], []

    # load all of the features
    features = load_features()
    x, pids = features['x'], features['pids']
    n, d = x.shape
    ptoi, itop = {}, {}
    for i, p in enumerate(pids):
        ptoi[p] = i
        itop[i] = p

    # construct the positive set
    y = np.zeros(n, dtype=np.float32)
    if pid:
        y[ptoi[pid]] = 1.0
    elif tags:
        tags_db = get_tags()
        tags_filter_to = tags_db.keys() if tags == 'all' else set(tags.split(','))
        for tag, pids in tags_db.items():
            if tag in tags_filter_to:
                for pid in pids:
                    y[ptoi[pid]] = 1.0

    if y.sum() == 0:
        return [], [], [] # there are no positives?

    # classify
    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=C)
    clf.fit(x, y)
    s = clf.decision_function(x)
    sortix = np.argsort(-s)
    pids = [itop[ix] for ix in sortix]
    scores = [100*float(s[ix]) for ix in sortix]

    # get the words that score most positively and most negatively for the svm
    ivocab = {v:k for k,v in features['vocab'].items()} # index to word mapping
    weights = clf.coef_[0] # (n_features,) weights of the trained svm
    sortix = np.argsort(-weights)
    words = []
    for ix in list(sortix[:40]) + list(sortix[-20:]):
        words.append({
            'word': ivocab[ix],
            'weight': weights[ix],
        })

    return pids, scores, words

def time_rank():
    mdb = get_metas()
    ms = sorted(mdb.items(), key=lambda kv: kv[1]['_time'], reverse=True)
    tnow = time.time()
    pids = [k for k, v in ms]
    scores = [(tnow - v['_time'])/60/60/24 for k, v in ms] # time delta in days
    return pids, scores

def random_rank():
    mdb = get_metas()
    pids = list(mdb.keys())
    shuffle(pids)
    scores = [0 for _ in pids]
    return pids, scores

def render_pid(pid):
    # render a single paper with just the information we need for the UI
    pdb = get_papers()
    tags = get_tags()
    thumb_path = 'static/thumb/' + pid + '.jpg'
    thumb_url = thumb_path if os.path.isfile(thumb_path) else ''
    d = pdb[pid]
    return dict(
        weight = 0.0,
        id = d['_id'],
        title = d['title'],
        time = d['_time_str'],
        authors = ', '.join(a['name'] for a in d['authors']),
        tags = ', '.join(t['term'] for t in d['tags']),
        utags = [t for t, pids in tags.items() if pid in pids],
        summary = d['summary'],
        thumb_url = thumb_url,
    )

def search_papers(rank='time', tags='', pid='', time_filter='', q='', skip_have='no', svm_c='', page_number=1):
    # if a query is given, override rank to be of type "search"
    # this allows the user to simply hit ENTER in the search field and have the correct thing happen
    if q:
        rank = 'search'

    # try to parse opt_svm_c into something sensible (a float)
    try:
        C = float(svm_c)
    except ValueError:
        C = 0.01 # sensible default, i think

    # rank papers: by tags, by time, by random
    words = [] # only populated in the case of svm rank
    if rank == 'search':
        pids, scores = search_rank(q=q)
    elif rank == 'tags':
        pids, scores, words = svm_rank(tags=tags, C=C)
    elif rank == 'pid':
        pids, scores, words = svm_rank(pid=pid, C=C)
    elif rank == 'time':
        pids, scores = time_rank()
    elif rank == 'random':
        pids, scores = random_rank()
    else:
        raise ValueError("rank %s is not a thing" % (rank, ))

    # filter by time
    if time_filter:
        mdb = get_metas()
        kv = {k:v for k,v in mdb.items()} # read all of metas to memory at once, for efficiency
        tnow = time.time()
        deltat = int(time_filter)*60*60*24 # allowed time delta in seconds
        keep = [i for i,pid in enumerate(pids) if (tnow - kv[pid]['_time']) < deltat]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    # optionally hide papers we already have
    if skip_have == 'yes':
        tags = get_tags()
        have = set().union(*tags.values())
        keep = [i for i,pid in enumerate(pids) if pid not in have]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    # crop the number of results to RET_NUM, and paginate
    try:
        page_number = max(1, int(page_number))
    except ValueError:
        page_number = 1
    start_index = (page_number - 1) * RET_NUM # desired starting index
    end_index = min(start_index + RET_NUM, len(pids)) # desired ending index
    pids = pids[start_index:end_index]
    scores = scores[start_index:end_index]

    # render all papers to just the information we need for the UI
    papers = [render_pid(pid) for pid in pids]
    for i, p in enumerate(papers):
        p['weight'] = float(scores[i])

    return papers

if __name__ == '__main__':
    papers = search_papers()
    print(papers)