from armozeen.parser.pipeline import PipelineStage
from armozeen.types import Expression, Token, Expressions, Tokens, UnaryOp, TypeDef, Block, Type, ArmozeenException
from armozeen.utils import check_expr, check_token, token_or_expr, find_ignore_whitespace, reverse_find_ignore_whitespace, iter_idented, split_by, item_is_whitespace


class UnbalancedTokenPair(ArmozeenException): pass


class TokenPairCollapser(PipelineStage):
    def __init__(self, ltoken, rtoken, terminators, typ):
        self.ltoken = ltoken
        self.rtoken = rtoken
        self.terminators = terminators
        self.type = typ

    def run(self, tokens):
        e, l, r = Expression(None, Expressions.Root), 0, 0
        
        for c in tokens:
            if isinstance(c, Expression):
                if c.type == Expressions.Root:
                    e.children += c.children
                    continue
                elif c.type not in [Expressions.String, 'sstring']:
                    c.children = self.run(c.children)
                e.children.append(c)
                continue

            if c.char == self.ltoken:
                l += 1
                new = Expression(e, self.type)
                e.children.append(new)
                e = new
            elif c.char == self.rtoken:
                e = e.parent
                r += 1
            elif c.char in self.terminators:
                if l > r:
                    break
                else:
                    e.children.append(c)
            else:
                e.children.append(c)

        if l != r:
            raise UnbalancedTokenPair('Unbalanced {} token pair'.format(self.type))

        return e.children


class Stringifier(PipelineStage):
    def __init__(self, token, type_=Expressions.String):
        self.token = token
        self.type = type_

    def run(self, tokens):
        e, tk = None, []
        for t in tokens:
            if isinstance(t, Expression):
                tk.append(t)
                continue
            if t.char == self.token:
                if (e
                   and e.children
                   and e.children[-1]
                   and isinstance(e.children[-1], Token)
                   and e.children[-1].char == '\\'):
                    e.children.pop()
                    e.children.append(t)
                elif e:
                    ch = e.children
                    e.children = [Token("".join([c.char for c in ch]), t.position)]
                    tk.append(e)
                    e = None
                else:
                    e = Expression(None, self.type)
            elif not e:
                tk.append(t)
            else:
                e.children.append(t)
        if e:
            tk.append(e)
        return tk


class WhitespaceRemover(PipelineStage):
    def run(self, items):
        toks = []
        for t in items:
            if isinstance(t, Expression):
                if check_expr(t, Expressions.Indentation):
                    continue
                t.children[:] = self.run(t.children)
                toks.append(t)
            elif isinstance(t, list):
                t[:] = self.run(t)
                toks.append(t)
            elif not check_token(t, [Tokens.Space, Tokens.Tab, '\0']):
                toks.append(t)
        return toks


class Consecutive(PipelineStage):
    def __init__(self, tok, typ):
        self.token = tok
        self.type = typ

    def run(self, items):
        toks, buff = [], []
        for t in items:
            value = t.char if isinstance(t, Token) else None
            if value == self.token:
                buff.append(t)
            else:
                if len(buff) > 0:
                    if len(buff) > 1:
                        exp = Expression(None, self.type)
                        exp.children.append(Token(str(len(buff))))
                        toks.append(exp)
                    else:
                        toks.append(buff[0])
                    buff = []
                toks.append(t)
        if len(buff) > 1:
            exp = Expression(None, self.type)
            exp.children.append(Token(str(len(buff))))
            toks.append(exp)
        elif len(buff) == 1:
            toks.append(buff[0])
        return toks


class PairwiseCollapser(PipelineStage):
    def __init__(self, pairs):
        self.pairs = pairs

    def run(self, items):
        last, toks = None, []
        for t in items:
            toks.append(t)

            value = t.char if isinstance(t, Token) else None

            for first, second, typ in self.pairs:
                if first == last and value == second:
                    toks = toks[:-2]
                    expr = Expression(None, typ)
                    toks.append(expr)

            last = value
        return toks


class Keywordizer(PipelineStage):
    def __init__(self, kwcls):
        self.keywords = kwcls

    def run(self, items):
        toks = []
        for t in items:
            value = t.char if isinstance(t, Token) else None
            if value in self.keywords.All:
                toks.append(Expression(None, value))
            else:
                toks.append(t)
        return toks


class BuiltinTyper(PipelineStage):
    def __init__(self, kwcls):
        self.keywords = kwcls

    def run(self, items):
        toks = []
        for t in items:
            value = t.char if isinstance(t, Token) else None
            if value in self.keywords.All:
                toks.append(Expression(None, Expressions.TypeName))
                toks[-1].children.append(t)
            else:
                toks.append(t)
        return toks


class Nameizer(PipelineStage):
    def run(self, items):
        toks = []
        for t in items:
            value = t.char if isinstance(t, Token) else ''
            if value and value not in Tokens.All:
                try:
                    value = str(int(value, 0))
                    toks.append(Expression(None, Expressions.Number))
                    toks[-1].children.append(Token(value, t.position))
                except:
                    toks.append(Expression(None, Expressions.Name))
                    toks[-1].children.append(Token(value, t.position))
            else:
                toks.append(t)
        return toks


class AngledExpressions(PipelineStage):
    """ Angled expressions are used in three distinct contexts: 
    
        Bit slicing:

          bits(6) variable = '000110';
          variable<4:5> == '11'

        Sub-structure selections and assignments:

          struct MyStruct {
             bit A,
             bit A1,
             bit B,
             bit B2,
             bit C,
          }

          MyStruct myStruct;
          myStruct.<A, B, C> = '101'

        Multiple variable assignments (MVA):

          bit A;
          bit B;
          bit C;
          
          <A, B, C> = '100'

        Any nesting of angled expressions is forbidden. Additional rules apply
        for each context:
          
          1) Bit slices can have only variables, bit strings, numbers and basic
             arithmetic operators in their body

          2) Sub-structure context and MVA context allow only names and commas in the body
    """
    @staticmethod
    def check_recursive(items):
        had_colon = False
        for item in items:
            if isinstance(item, Token):
                if item.char not in [Tokens.Colon, Tokens.Star, Tokens.Plus, Tokens.Minus, 
                                     Tokens.Comma, Tokens.Space]:
                    return False
            elif isinstance(item, Expression):
                if item.type == Expressions.Paren:
                    return AngledExpressions.check_recursive(item.children)

                if item.type not in [Expressions.Name, Expressions.Number, Expressions.Division,
                                     Expressions.Modulo, Expressions.String]:
                    return False
        return True

    @staticmethod
    def check_substruct(items, first=True):
        for item in items:
            if isinstance(item, Token) and item.char not in [Tokens.Comma, Tokens.Space]:
                return False
            if isinstance(item, Expression) and item.type != Expressions.Name:
                return False
        return True

    def run(self, items):
        toks, skip = [], 0
        for i, t in enumerate(items):
            if i < skip:
                continue
            value = t.char if isinstance(t, Token) else ''
            if value == '<':
                for i2, t2 in enumerate(items[i:]):
                    value2 = t2.char if isinstance(t2, Token) else ''
                    if value2 in ['\n', ';']:
                        toks.append(t)
                        break
                    if value2 == '>':
                        its = items[i+1:i+i2]
                        on_name = check_expr(items[i-1], [Expressions.Name, Expressions.Bracket])
                        if AngledExpressions.check_recursive(its) and on_name:
                            skip = i+i2+1
                            exp = Expression(None, Expressions.Bitselect)
                            exp.children += its
                            toks.append(exp)
                            break
                        else:
                            valid_prev_tokens = [Tokens.Dot, Tokens.NewLine, Tokens.Comma, Tokens.Space]
                            substruct = isinstance(items[i-1], Token) and items[i-1].char in valid_prev_tokens
                            if substruct and AngledExpressions.check_substruct(its):
                                toks.pop()
                                skip = i+i2+1
                                exp = Expression(None, Expressions.Substruct)
                                exp.children += its
                                toks.append(exp)
                                break
                            else:
                                toks.append(t)
                                break
            else:
                if isinstance(t, Expression):
                    t.children[:] = self.run(t.children)
                toks.append(t)
        return toks


class CommentRemover(PipelineStage):
    def run(self, items):
        toks, skip, depth = [], 0, 0
        for i, t in enumerate(items):
            if i < skip:
                continue
            depth = 0
            if check_expr(t, Expressions.Comment):
                for k, r in enumerate(items[i+1:]):
                    if check_token(r, '\n'):
                        skip = i+k+1
                        break
            elif check_expr(t, Expressions.MultiLineStart):
                for k, r in enumerate(items[i+1:]):
                    if check_expr(r, Expressions.MultiLineStart):
                        depth += 1
                    elif check_expr(r, Expressions.MultiLineEnd):
                        if depth == 0:
                            skip = i+k+2
                            break
                        else:
                            depth -= 1
            else:
                toks.append(t)

        if depth > 0:
            raise ValueError('Unbalanced `/*` `*/` comment pair')

        return toks


class FindUnaryOperators(PipelineStage):
    def run(self, items):
        toks, skip = [], False
        for i, t in enumerate(items):
            if skip:
                skip = False
                continue
            value = t.char if isinstance(t, Token) else ''
            if value == Tokens.Exclamation:
                if len(items) > (i + 1):
                    skip = True
                    toks.append(UnaryOp(Expressions.BoolNot, items[i + 1]))
                else:
                    raise DanglingTokenError(value)
            elif value == Tokens.Minus:
                bad_prev = [Tokens.Comma, Tokens.LessThan, Tokens.GreaterThan]
                if i == 0 or (isinstance(items[i - 1] , Token) and items[i - 1].char in bad_prev):
                    if len(items) > (i + 1):
                        skip = True
                        toks.append(UnaryOp(Expressions.Negation, items[i + 1]))
                    else:
                        toks.append(t)
                        #raise DanglingTokenError(value)
                else:
                    toks.append(t)
            else:
                if isinstance(t, Expression):
                    t.children[:] = self.run(t.children)
                toks.append(t)
        return toks


class HandleBitsType(PipelineStage):
    @staticmethod
    def check_bits_tokens(tokens):
        def checker(item):
            valid_tokens = [Tokens.Star, Tokens.Plus, Tokens.Minus, Tokens.Space]
            valid_expressions = [Expressions.Name, Expressions.Number, Expressions.Division,
                                 Expressions.Modulo, Expressions.XOR, Expressions.Paren]
            return token_or_expr(item, valid_tokens, valid_expressions)

        return all(checker(t) for t in tokens)

    def run(self, items):
        toks, skip_next = [], False
        for i, t in enumerate(items):
            if skip_next:
                skip_next = False
                continue

            if check_expr(t, Expressions.Name) and check_token(t.children[0], 'bits'):
                if len(items) < i:
                    raise Exception('Not enouugh tokens for bits')
                if not check_expr(items[i + 1], 'paren'):
                    raise Exception('Expected bits size, found' + str(t))
                parens = items[i + 1]
                if not HandleBitsType.check_bits_tokens(parens.children):
                    raise Exception('Invalid token in bits length:\n'+str(parens.children))
                toks.append(Type('bits', parens.children))
                skip_next = True
            elif check_expr(t, Expressions.Name) and check_token(t.children[0], 'bit'):
                toks.append(Type('bits', 1))
            else:
                if isinstance(t, Expression):
                    t.children[:] = self.run(t.children)
                toks.append(t)
        return toks


class FindTypes(PipelineStage):
    def run(self, items):
        toks, skip = [], 0
        for i, t in enumerate(items):
            if i < skip:
                continue
            if check_expr(t, Expressions.Type):
                #if not (i == 0 or check_token(items[i - 1], [Tokens.NewLine, '\0'])):
                #    toks.append(t)
                #    continue

                n, name = find_ignore_whitespace(items[i + 1:])
                if not check_expr(name, Expressions.Name):
                    toks.append(t)
                    continue

                m, isop = find_ignore_whitespace(items[i + n + 2:])
                if check_token(isop, Tokens.Semicolon):
                    toks.append(TypeDef(name, None))
                    skip = i + n + m + 3;
                    continue

                if not check_expr(isop, Expressions.Is):
                    toks.append(t)
                    continue

                b, paren = find_ignore_whitespace(items[i + n + m + 3:])
                if not check_expr(paren, Expressions.Paren):
                    toks.append(t)
                    continue

                paren.type = 'fields'
                newchildren = []
                for typ_, nam_ in split_by(paren.children, Tokens.Comma, sanitize=True):
                    newchildren.append(Expression(None, 'field'))
                    newchildren[-1].children = [typ_, nam_]
                paren.children = newchildren
                toks.append(TypeDef(name, paren))
                skip = i + n + m + b + 4
            else:
                toks.append(t)
        return toks


class FindTuples(PipelineStage):
    def run(self, items):
        toks = []
        for i, t in enumerate(items):
            valid_tokens = [Tokens.Comma, Tokens.Minus, Tokens.Space]
            if check_expr(t, Expressions.Paren) and all(token_or_expr(c, valid_tokens, Expressions.Name) for c in t.children):
                prev_is_name = check_expr(items[i - 1], [Expressions.Name, Expressions.Is, 'type'])
                if i == 0 or not prev_is_name:
                    t.type = Expressions.Tuple
                t.children[:] = self.run(t.children)
                toks.append(t)
            else:
                if isinstance(t, Expression):
                    t.children[:] = self.run(t.children)
                toks.append(t)
        return toks


class FindTuplesLateStage(PipelineStage):
    ''' Convert all outstanding `paren` expressions to `tuple` expressions '''
    def run(self, items):
        toks = []
        for i, t in enumerate(items):
            if check_expr(t, Expressions.Paren) and len(t.children) > 1:
                t.type = 'tuple'
                t.children[:] = self.run(t.children)
                toks.append(t)
            else:
                if isinstance(t, Expression):
                    t.children[:] = self.run(t.children)
                elif isinstance(t, list):
                    t[:] = self.run(t)
                toks.append(t)
        return toks


class FindAssignment(PipelineStage):
    def run(self, items):
        toks = []
        for i, t in enumerate(items):
            if check_token(t, Tokens.Equals):
                _, prev = reverse_find_ignore_whitespace(items[:i])
                invalid_expressions = [Expressions.Name, Expressions.Tuple, Expressions.Bracket,
                                       Expressions.Bitselect, Expressions.Substruct, Expressions.Paren]
                if not check_expr(prev, invalid_expressions):
                    toks.append(t)
                else:
                    toks.append(Expression(None, Expressions.Assignment))
            else:
                self.maybe_recurse(t)
                toks.append(t)
        return toks


class FindFunctionDefinitions(PipelineStage):
    def run(self, items):
        toks, skip_next = [], False
        for i, t in enumerate(items):
            if skip_next:
                skip_next = False
                continue
            if check_expr(t, Expressions.Name):
                fndef = (i+1) < len(items)
                fndef &= check_expr(items[i+1], Expressions.Paren)
                short = False
                k, prev = reverse_find_ignore_whitespace(items[:i])
                return_values = [Expressions.TypeName, Expressions.Tuple, Expressions.Name, Expressions.Paren, Expressions.Bits, Expressions.TypeExp]

                if check_token(prev, [Tokens.NewLine, Tokens.Nil]):
                    short = True
                    _, ind = find_ignore_whitespace(items[i+2:])
                    fndef = check_token(ind, Tokens.NewLine)
                elif check_expr(prev, return_values):
                    _, prev_prev = reverse_find_ignore_whitespace(items[i-(k+1):])
                    fndef &= check_token(prev_prev, [Tokens.NewLine, Tokens.Nil])
                else:
                    fndef = False

                if not fndef:
                    toks.append(t)
                else:
                    e = Expression(None, Expressions.Fundef)
                    # ReturnType, Name, Arguments

                    def clean_recursive(itm):
                        newitems = []
                        for subitem in itm:
                            if not item_is_whitespace(subitem):
                                newitems.append(subitem)
                        return newitems

                    cleaned = clean_recursive(items[i + 1].children)

                    params = []
                    for typ, name in split_by(cleaned, Tokens.Comma):
                        params.append(Expression(None, Expressions.Parameter))
                        params[-1].children += [typ, name]

                    items[i+1].children = params
                    items[i+1].type = Expressions.Parameters

                    e.children = [prev if not short else None, t, items[i+1]]
                    skip_next = True
                    for _ in range(k+1):
                       toks.pop()
                    toks.append(e)
            else:
                toks.append(t)
        return toks


class FindFunctionCalls(PipelineStage):
    def run(self, items):
        toks, skip_next = [], False
        for i, t in enumerate(items):
            if skip_next:
                skip_next = False
                continue
            if check_expr(t, [Expressions.Name, Expressions.Load]):
                if (i+1) < len(items) and check_expr(items[i+1], Expressions.Paren):
                    toks.append(Expression(None, Expressions.Funcall))
                    toks[-1].children = [t, items[i+1]]
                    # change name from paren for late stage tuple recognition
                    items[i+1].type = 'pparen'
                    skip_next = True
                else:
                    toks.append(t)
            else:
                self.maybe_recurse(t)
                toks.append(t)
        return toks


class FindInlineConditionals(PipelineStage):
    def run(self, items):
        toks, skip_until = [], -1
        for i, t in enumerate(items):
            if i < skip_until:
                continue
            stop = False
            condition, if_, else_ = [], [], []
            term_tokens = [Tokens.NewLine, Tokens.Semicolon]
            if check_expr(t, Expressions.If):
                for k, tt in enumerate(items[i:]):
                    if check_token(tt, term_tokens):
                        # TODO: RAISE?
                        break
                    if stop: break
                    if check_expr(tt, Expressions.Then):
                        condition = items[i+1:i+k]
                        for n, ttt in enumerate(items[i+k+1:]):
                            if check_token(ttt, term_tokens):
                                # TODO: RAISE?
                                #
                                # REAL TODO: Turn this piece of shit code into grammar
                                stop = True
                                break
                            if stop: break
                            if check_expr(ttt, Expressions.Else):
                                if_ = items[i+k+1:i+k+n+1]
                                last = 0
                                for m, tttt in enumerate(items[i+k+n+2:]):
                                    last = m
                                    if check_token(tttt, term_tokens):
                                        stop = True
                                        break
                                else_ = items[i+k+n+2:i+k+n+last+2]
                                skip_until = i+k+n+last+3
            if condition and if_ and else_:
                toks.append(Expression(None, Expressions.InlineIf))
                c = Expression(None, Expressions.Condition)
                c.children = condition
                ifb = Expression(None, Expressions.Branch)
                ifb.children = if_
                elseb = Expression(None, Expressions.Branch)
                elseb.children = else_
                toks[-1].children = [c, ifb, elseb]
            else:
                self.maybe_recurse(t)
                toks.append(t)
        return toks


class IndentationCollapser(PipelineStage):
    def run(self, items):
        ''' Collapse indented lines into blocks with N+ level of indentation

            The function splits the incoming token stream by the new line character
            and collapses fitting consecutive blocks. After processing the new line
            tokens are re-inserted back for further expression parsing.
        '''
        root = block = Block(0, None)
        for n, l in iter_idented(items): 
            idx = 1 if check_expr(l[0], Expressions.Indented) else 0
            if n == block.level:
                block.children += l[idx:] + [Token(Tokens.NewLine)]
            elif n > block.level:
                nblock = Block(n, block, l[idx:] + [Token(Tokens.NewLine)])
                block.children.append(nblock)
                block = nblock
            else:
                while block.level != n:
                    block = block.parent
                    if not block:
                        raise Exception("Bad indtentation")
                block.children += l[idx:] + [Token(Tokens.NewLine)]
                #block.children.append(l[idx:])
        return [root]


class ParenNewlineRemover(PipelineStage):
    ''' Clean up new line characters from parentheses  '''
    def run(self, items):
        toks = []
        for t in items:
            if check_expr(t, [Expressions.Paren, Expressions.Curly, Expressions.Bracket]):
                t.children = [r for r in t.children if not check_token(r, Tokens.NewLine)]

            self.maybe_recurse(t)
            toks.append(t)
        return toks


class CleanupStage(PipelineStage):
    ''' Remove superfluous indentation and some other junk '''
    def run(self, items):
        toks = []
        for t in items:
            if isinstance(t, Expression):
                if check_expr(t, Expressions.Indentation):
                    continue
                t.children[:] = self.run(t.children)
                toks.append(t)
            elif isinstance(t, list):
                t[:] = self.run(t)
                toks.append(t)
            elif not check_token(t, [Tokens.Space, Tokens.Tab, '\0', '\n', ';', ',']):
                toks.append(t)
        return toks


class NameLoads(PipelineStage):
    ''' Early resolution of loads like:

          pet.name
          phone.device.ring()
    '''
    def run(self, items):
        toks, skip = [], -1
        for i, t in enumerate(items):
            if skip > i:
                continue
        
            if (i + 1) < len(items) and toks and check_expr(toks[-1], 'load') and check_token(t, Tokens.Dot) and check_expr(items[i + 1], 'name'):
                x = toks.pop()
                toks.append(Expression(None, 'load'))
                toks[-1].children = [x, Token(Tokens.Comma), items[i + 1]]
                skip = i + 2
                continue

            if check_expr(t, Expressions.Name):
                n, dot = find_ignore_whitespace(items[i + 1:])
                if not check_token(dot, Tokens.Dot):
                    toks.append(t)
                    continue

                m, loaded = find_ignore_whitespace(items[i + n + 2:])
                if not check_expr(loaded, Expressions.Name):
                    toks.append(t)
                    continue

                skip = i + n + m + 3
                toks.append(Expression(None, 'load'))
                toks[-1].children = [t, Token(Tokens.Comma), loaded]
            else:
                toks.append(t)
        return toks
