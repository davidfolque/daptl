import shelve
import pandas as pd
from filelock import FileLock


def construct_query(d):
    query = ''
    for key, value in d.items():
        query += key + '=='
        if type(value) == str:
            query += '"' + value + '"'
        else:
            query += str(value)
        query += ' and '
    return query[:-5]


class Persistence:
    def __init__(self, path):
        self.path = path
        self.lock = FileLock(self.path + '.lock')
        self.sh = None
        
    def __enter__(self):
        self.lock.acquire()
        self.sh = shelve.open(self.path)
        return self
        
    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.sh.close()
        self.lock.release()
        self.sh = None    
    
    def get_index(self):
        return self.sh.get('index', pd.DataFrame())
    
    def get_persistence_ids(self, config):
        index = self.get_index()
        if len(index) > 0:
            query = construct_query(config)
            return index.query(query).persistence_id.tolist()
        return []
    
    def get_entries(self, config):
        persistence_ids = self.get_persistence_ids(config)
        return [self.sh['item' + str(pid)] for pid in persistence_ids]
    
    def add_new_entry(self, config, contents):
        # Fetch next id, or start at 0.
        next_id = self.sh.get('next_id', 0)
        id_str = 'item' + str(next_id)
        
        # Add entry to configs.
        index = self.get_index()
        index = index.append(pd.DataFrame({'persistence_id': next_id, **config}, index=[0]), 
                             ignore_index=True)
        self.sh['index'] = index
        
        # Add contents.
        self.sh[id_str] = contents
        
        # Update next_id.
        self.sh['next_id'] = next_id + 1
        
        
if __name__ == '__main__':
    
    print('Hello')
    with Persistence('test.db') as db:
    
        config = {'batch_size': 64, 'lr': 0.1, 'mode':'default'}
        print(db.get_persistence_ids(config))
        print(db.get_entries(config))
        
        db.add_new_entry(config, 12345)
        print(db.sh['index'].dtypes)
        
        print(db.get_persistence_ids(config))
        print(db.get_entries(config))
        
        db.add_new_entry(config, {'results': 'hello'})
        
        print(db.get_persistence_ids(config))
        print(db.get_entries(config))
        
        config['mode'] = 'random'
        db.add_new_entry(config, {'results': 'even better', 'acc': 1000})
        
        print(db.get_persistence_ids(config))
        print(db.get_entries(config))
        
        del config['mode']
        print(db.get_entries(config))

























        
