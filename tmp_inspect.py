import pickle, pathlib
p = pathlib.Path('models/final_model_state.pkl')
print('exists', p.exists())
with p.open('rb') as f:
    obj = pickle.load(f)
print('type', type(obj))
if isinstance(obj, dict):
    print('keys', list(obj.keys()))
    for k,v in obj.items():
        t = type(v)
        shape = getattr(v, 'shape', None)
        print(k, t, shape)
else:
    print('obj attrs', [a for a in dir(obj) if not a.startswith('_')][:40])
