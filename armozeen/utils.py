from armozeen.types import Expression, Token, Expressions, Tokens, LanguageStage


def replace(seq, item, with_):
    ''' Replace `item` in `seq` with `with_`

    '''
    for i in seq:
        if i == item:
            yield with_
        else:
            yield i


def check_wildcard(et, instr):
    ''' Check if `et` == `instr` and "expand" wildcards

    '''
    idx = et.find('*')
    if idx >= 1:
        return instr[:idx] == et[:idx]
    return et == instr


def item_is_whitespace(item):
    ''' Check if `item` is whitespace token or expression

    '''
    if isinstance(item, Token):
        return item.char in [Tokens.Space, Tokens.Tab]
    if isinstance(item, Expression):
        return item.type in Expressions.Indentation
    return False


def reverse_find_ignore_whitespace(items):
    ''' Reverse-find first non-whitespace expression or token in `items`

    '''
    for i, t in enumerate(reversed(items)):
        if item_is_whitespace(t):
            continue
        return i, t
    return -1, None


def find_ignore_whitespace(items):
    ''' Find first non-whitespace expression or token in `items`

    '''
    for i, t in enumerate(items):
        if item_is_whitespace(t):
            continue
        return i, t
    return -1, None


def check_ast(e=None, t=None):
    if not t or not e:
        return False
    if not isinstance(t, list):
        t = [t]
    from armozeen.ast import AstNode
    return e and isinstance(e, AstNode) and any(e.type.endswith(t_) for t_ in t)


def check_expr(e=None, t=None):
    if not t or not e:
        return False
    if not isinstance(t, list):
        t = [t]
    return e and isinstance(e, Expression) and any(check_wildcard(t_, e.type) for t_ in t)


def check_token(t=None, v=None):
    if not v or not v:
        return False
    if not isinstance(v, list):
        v = [v]
    return t and isinstance(t, Token) and t.char in v


def token_or_expr(x, values=None, expr_types=None):
    if not (values or expr_types):
        raise ValueError('either values or expr_types has to be provided')
    return check_token(x, values) or check_expr(x, expr_types)


def find_expr_ctx(items, exprt, ctx=None, ctr=0):
    ''' Find expression type `exprt` in `items` and optionally return
        more items than just the matching expression dependending on context `ctx` tuple:

          find_expr_ctx(items, Expressions.Add, ctx=(-1, 2))

        This returns a list containing the matching expression, the preceding expression (-1)
        and next 2 expresisons (2).

        :param items: List of tokens and expressions
        :type items:  list[Expression, Token]
        :param exprt: Expression type
        :type exprt:  str
        :param ctx:   Return window context
        :type exprt:  tuple(lowerbound, upperbound)
        :param ctr:   Start counting from this number
        :type ctr:    int
        :returns:     Tuple with offset of the found item and matching item(s)
        :rtype:       tuple(offset, item(s))
    '''
    l = len(items)
    for i, t in enumerate(items):
        if check_expr(t, exprt):
            if ctx:
                lb, ub = ctx
                ilb, iub = i + lb, i + ub
                low = ilb if ilb > 0 else 0
                high = iub if iub < l else l-1
                return (ctr + i, items[low:high])
            return (ctr + i, t)


def split_by(items, sep, sanitize=False):
    ''' Split `items` by `sep`

        :param items: List of tokens and expresisons
        :type  items: list[Expression, Token]
        :param sep:   Separator token
        :type  sep:   str
    '''
    toks, cand = [], []
    for t in items:
        if sanitize and (item_is_whitespace(t) or check_token(t, Tokens.NewLine)):
            continue
        if check_token(t, sep):
            if cand:
                toks.append(cand)
                cand = []
        else:
            cand.append(t)
    if cand:
        toks.append(cand)
    return toks


def get_indentation_entries(items):
    ''' Get indentation levels from input sequence `items` '''
    for t in items:
        if check_expr(t[0], Expressions.Indented):
            yield (int(t[0].children[0].char), t)
        else:
            yield (0, t)


def pairwise(items):
    ''' Pairwise iterate the input sequence `items`  '''
    i = iter(items)
    while True:
        yield next(i), next(i)


def iter_idented(items):
    ''' Get indentation levels per each line '''
    return get_indentation_entries(split_by(items, Tokens.NewLine))


def wildcard(item):
    ''' Decorate input item with a wildcard character '''
    return item + '*'


def swap((f, s)):
    ''' Swap first and second tuple elements  '''
    return s, f


def nest(o, *addr):
    ''' Nest into expression `o` children nodes according to indices in addr, e.g.:

         a = o.children[0].children[1].children[2].children[1]
         b = nest(o, 0, 1, 2, 1)

        Such as `a == b`
    '''
    for a in addr:
        o = o.children[a]
    return o


def execute(stage, code, **kwargs):
    from armozeen.parser.pipeline import Pipeline
    from armozeen.parser.pipeline.lexemes import Lexemes
    from armozeen.parser.pipeline.lexer import CleanupStage, FindTuplesLateStage
    from armozeen.parser.pipeline.tokenizer import Tokenizer
    from armozeen.interpeter.run import Run
    from armozeen.interpeter import AstBuilder
    from armozeen.parser.grammars import get_grammars_flat

    pipeline = []

    pipeline.append(Tokenizer(Tokens.All))
    pipeline.extend(Lexemes)

    if stage > LanguageStage.PreGrammars:
        pipeline.extend(get_grammars_flat())

    pipeline.append(FindTuplesLateStage())

    if stage >= LanguageStage.AST:
        pipeline.append(AstBuilder())
        pipeline.append(CleanupStage())

    if stage >= LanguageStage.Interpret:
        pipeline.append(Run(**kwargs))

    p = Pipeline(pipeline)

    return p.run(code)