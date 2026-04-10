import sys, os, pickle, numpy as np, pathlib
print('executable:', sys.executable)
print('numpy version:', np.__version__)
print('cwd:', os.getcwd())
for f in ['models/sentiment_model.pkl', 'models/rating_model.pkl', 'models/tfidf_vectorizer.pkl']:
    print(f, 'exists:', os.path.exists(f))
    if os.path.exists(f):
        try:
            with open(f, 'rb') as fh:
                obj = pickle.load(fh)
            print('loaded', f, 'type', type(obj))
        except Exception as e:
            print('failed to load', f, e)
