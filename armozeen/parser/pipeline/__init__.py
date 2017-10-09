from armozeen.types import Expression

class PipelineStage(object):
    ''' Simple interface for creating pipelines '''
    def run(self, items):
        raise NotImplementedError

    def maybe_recurse(self, t):
        ''' Helper function to simplify life  '''
        if isinstance(t, Expression):
            t.children[:] = self.run(t.children)


class Pipeline(PipelineStage):
    ''' Pipeline is a collection of multiple `PipelineStage` objects
        executed sequentally, passing output of one stage as an input
        to the next
    '''
    def __init__(self, items):
        self.items = items

    def run(self, items):
        r = items
        for item in self.items:
            #print('** Running: {:24} <0x{:x}>'.format(item.__class__.__name__, id(item)))
            r = item.run(r)
        return r
