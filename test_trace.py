import sys
class Trace(object):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch':
            import traceback
            traceback.print_stack()
            sys.exit(1)
        return None
sys.meta_path.insert(0, Trace())
import api.index
