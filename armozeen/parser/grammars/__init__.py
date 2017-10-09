from armozeen.parser.pipeline.grammar import Grammar, GExpr, GToken, GConsume, GConsumeT
from armozeen.types import Tokens, Token, Expression, Expressions, Type, TypedVariable
from armozeen.utils import check_expr, check_token, token_or_expr, wildcard as W
from collections import defaultdict


LineTerm = GToken(['\n', ';'])
NlTerm = GToken('\n')


_registered_grammars = defaultdict(list)


def register_grammar(stage, sort):
    ''' Register grammar in given `stage` under `sort` sort key '''

    def outer(C):

        _registered_grammars[stage].append((sort, C()))

        def inner(*args, **kwargs):
            return C

    return outer


def get_grammars_flat():
    ''' Gets flat grammar list sorted by (stage, key) '''

    stages = ('stage1', 'numeric', 'stage2', 'logical', 'stage3')
    out = []
    for s in stages:
        out += map(lambda (_, b): b, _registered_grammars[s])

    return out


@register_grammar('stage2', 7)
class IfStatementGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.IfThen), GConsumeT(0, [W(Expressions.Elsif), W(Expressions.Else)])]

    def create_item(self, tokens, items, consumed):
        block = items[0]
        condition = consumed[0]
        tokens.append(Expression(None, Expressions.IfStatement))
        block.children[:] = self.run(block.children)
        tokens[-1].children = [block] + condition


@register_grammar('stage1', 0)
class IfThenGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.If), GConsume(0), GExpr(Expressions.Then), NlTerm, GExpr(Expressions.Block)]

    def create_item(self, tokens, items, consumed):
        block = items[-1]
        condition = consumed[0]
        tokens.append(Expression(None, Expressions.IfThen))
        block.children[:] = self.run(block.children)
        tokens[-1].children = [condition, block]


@register_grammar('stage1', 3)
class IfThenSimpleGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.If), GConsume(0), GExpr(Expressions.Then), GExpr(Expressions.Return), GConsume(1), LineTerm]

    def create_item(self, tokens, items, consumed):
        condition = consumed[0]
        branch = consumed[1]
        tokens.append(Expression(None, 'ifthensimple'))
        tokens[-1].children = [condition, branch]


@register_grammar('stage1', 4)
class ElsifThenSimpleGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.Elsif), GConsume(0), GExpr(Expressions.Then), GExpr(Expressions.Return), GConsume(1), LineTerm]

    def create_item(self, tokens, items, consumed):
        condition = consumed[0]
        branch = consumed[1]
        tokens.append(Expression(None, 'elfsifthensimple'))
        tokens[-1].children = [condition, branch]


@register_grammar('stage1', 5)
class ElseSimpleGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.Else), GExpr(Expressions.Return), GConsume(0), LineTerm]

    def create_item(self, tokens, items, consumed):
        branch = consumed[0]
        tokens.append(Expression(None, 'elsesimple'))
        tokens[-1].children = [branch]


@register_grammar('stage1', 1)
class ElsifThenGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.Elsif), GConsume(0), GExpr(Expressions.Then), NlTerm, GExpr(Expressions.Block)]

    def create_item(self, tokens, items, consumed):
        block = items[-1]
        condition = consumed[0]
        tokens.append(Expression(None, 'elsifthen'))
        block.children[:] = self.run(block.children)
        tokens[-1].children = [condition, block]


@register_grammar('stage1', 2)
class ElseGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.Else), NlTerm, GExpr(Expressions.Block)]

    def create_item(self, tokens, items, consumed):
        block = items[-1]
        tokens.append(Expression(None, 'elseblock'))
        block.children[:] = self.run(block.children)
        tokens[-1].children = [block]


@register_grammar('stage1', 6)
class EnumGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.Enumeration), GConsume(0), GExpr(Expressions.Curly), LineTerm]

    def create_item(self, tokens, items, consumed):
        def check(ch):
            invalid_tokens = [Tokens.Space, Tokens.Tab, Tokens.NewLine]
            return not token_or_expr(ch, invalid_tokens, Expressions.Indentation)

        name = consumed[0]
        enum_items = items[-2]
        enum_items = list(filter(check, enum_items.children))
        tokens.append(Expression(None, Expressions.Enum))
        tokens[-1].children = [name.pop(), enum_items]


class BinaryOpGrammar(Grammar):
    def __init__(self, op_type, items):
        self.op_type = op_type
	self.items = items

    def create_item(self, tokens, items, consumed):
        a = items[-3]
        b = items[-1]
        tokens.append(Expression(None, self.op_type))
        tokens[-1].children = [a, None, b]


@register_grammar('stage2', 0)
class CaseOfGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.Case), GConsume(0), GExpr(Expressions.Of), NlTerm, GExpr(Expressions.Block)]

    def create_item(self, tokens, items, consumed):
        block = items[-1]
        condition = consumed[0]
        tokens.append(Expression(None, 'caseof'))
        block.children[:] = self.run(block.children)
        tokens[-1].children = [condition, block]


@register_grammar('stage2', 1)
class WhenGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.When), GExpr([Expressions.Name, Expressions.Number, Expressions.String, Expressions.Bitselect])]

    def create_item(self, tokens, items, consumed):
        condition = items[-1]
        tokens.append(Expression(None, 'whencase'))
        tokens[-1].children = [condition]


@register_grammar('stage2', 6)
class OtherwiseEarlyGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.Otherwise), GConsume(0), LineTerm]

    def create_item(self, tokens, items, consumed):
        condition = consumed[0]
        tokens.append(Expression(None, 'otherwiseearly'))
        tokens[-1].children = items[0].children + condition


@register_grammar('stage2', 5)
class OtherwiseBlockGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(Expressions.Otherwise), NlTerm, GExpr(Expressions.Block)]

    def create_item(self, tokens, items, consumed):
        block = items[-1]
        tokens.append(Expression(None, 'otherwiseblock'))
        block.children[:] = self.run(block.children)
        tokens[-1].children = items[0].children + [block]


@register_grammar('stage2', 3)
class WhenCaseEarlyGrammar(Grammar):
    @property
    def items(self):
        return [GExpr('whencase'), GConsume(0), LineTerm]

    def create_item(self, tokens, items, consumed):
        condition = consumed[0]
        tokens.append(Expression(None, 'whencaseearly'))
        tokens[-1].children = items[0].children + condition


@register_grammar('stage2', 2)
class WhenCaseBlockGrammar(Grammar):
    @property
    def items(self):
        return [GExpr('whencase'), NlTerm, GExpr('block')]

    def create_item(self, tokens, items, consumed):
        condition = consumed[0]
        block = items[-1]
        tokens.append(Expression(None, 'whencaseblock'))
        block.children[:] = self.run(block.children)
        tokens[-1].children = items[0].children + condition + [block]


@register_grammar('stage2', 10)
class NameBitselectGrammar(Grammar):
    @property
    def items(self):
        return [GExpr([Expressions.Name, Expressions.Load, 'bracketfunc']), GExpr(Expressions.Bitselect)]

    def create_item(self, tokens, items, consumed):
        tokens.append(Expression(None, Expressions.Bitselect))
        tokens[-1].children = [items[0], items[1].children]


@register_grammar('stage2', 9)
class BracketFuncGrammar(Grammar):
    @property
    def items(self):
        return [GExpr([Expressions.Name, Expressions.Load]), GExpr(Expressions.Bracket)]

    def create_item(self, tokens, items, consumed):
        tokens.append(Expression(None, 'bracketfunc'))
        tokens[-1].children = [items[0], items[1].children]


@register_grammar('stage2', 11)
class FoldConstGrammar(Grammar):
    @property
    def items(self):
        return [GExpr('constant'), GExpr([Expressions.Bits, 'name', 'typename', 'bit', 'integer', 'real', 'typed_variable'])]

    def create_item(self, tokens, items, consumed):
        size = 0
        if check_expr(items[1], ['typename', 'name']):
            name = items[1]
        else:
            name = items[1]
        if check_expr(items[1], Expressions.Bits):
            size = items[1].children[0].children[0]
        tokens.append(Type(name, size=size, const=True))
        tokens[-1].children = [items[0], items[1]]


@register_grammar('stage2', 12)
class TypedVariableGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(['typename', 'typeexp', 'name', 'bits', 'bit', 'integer', 'real']), GExpr('name')]

    def create_item(self, tokens, items, consumed):
        tokens.append(TypedVariable(items[0], items[-1]))
        tokens[-1].children = [items[0], items[-1]]


@register_grammar('stage3', 2)
class TypedRefVariableGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(['typename', 'typeexp', 'name', 'bits', 'bit', 'integer', 'real']), GToken('&'), GExpr('name')]

    def create_item(self, tokens, items, consumed):
        tokens.append(TypedVariable(items[0], items[-1], is_reference=True))
        tokens[-1].children = [items[0], items[-1]]


@register_grammar('stage2', 8)
class ForLoopGrammar(Grammar):
    @property
    def items(self):
        return [GExpr('for'), GExpr(Expressions.Name), GExpr(Expressions.Assignment), 
                GExpr([Expressions.Name, Expressions.Number, Expressions.Paren, 'op_num_*']), 
                GExpr(['to', 'downto']), GConsume(0), NlTerm, GExpr(Expressions.Block)]

    def create_item(self, tokens, items, consumed):
        init = items[1:4]
        block = items[-1]
        dest = consumed[0]
        tokens.append(Expression(None, 'forloop'))
        block.children[:] = self.run(block.children)
        tokens[-1].children = [init, dest, block]


@register_grammar('stage2', 13)
class BracketFundefGrammar(Grammar):
    @property
    def items(self):
        return [GExpr([Expressions.Bits, Expressions.TypeExp, 'load']), GExpr('bracketfunc'), NlTerm, GExpr(Expressions.Block)]

    def create_item(self, tokens, items, consumed):
        tokens.append(Expression(None, 'bracketfundef'))
        tokens[-1].children = [items[0], items[1], items[-1]]


@register_grammar('stage2', 4)
class FundefGrammar(Grammar):
    @property
    def items(self):
        return [GExpr('fundef'), NlTerm, GExpr('block')]

    def create_item(self, tokens, items, consumed):
        tokens.append(Expression(None, 'function_definition'))
        tokens[-1].children = [[items[0].children[0]]] + [[items[0].children[1]]] + [[items[0].children[2]]] + [items[-1]]


@register_grammar('stage3', 3)
class AssertGrammar(Grammar):
    @property
    def items(self):
        return [GExpr('assert'), GConsume(0), GToken(';')]

    def create_item(self, tokens, items, consumed):
        tokens.append(Expression(None, 'assertion'))
        tokens[-1].children = consumed[0]


@register_grammar('stage3', 4)
class ReturnGrammar(Grammar):
    @property
    def items(self):
        return [GExpr('return'), GExpr(), LineTerm]

    def create_item(self, tokens, items, consumed):
        tokens.append(Expression(None, 'return_value'))
        tokens[-1].children = [items[1]]


@register_grammar('stage2', 16)
class WriteBracketFundefGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(['bracketfunc']), GExpr('assignment'), GExpr('typed_variable'), NlTerm, GExpr(Expressions.Block)]

    def create_item(self, tokens, items, consumed):
        tokens.append(Expression(None, 'wbracketfundef'))
        tokens[-1].children = [items[0], items[2], items[-1]]


@register_grammar('stage2', 17)
class SubstructGrammar(Grammar):
    @property
    def items(self):
        return [GExpr(['name', 'load']), GExpr('substruct')]

    def create_item(self, tokens, items, consumed):
        tokens.append(Expression(None, 'destructure'))
        tokens[-1].children = [items[0], items[1].children]


@register_grammar('stage3', 5)
class ArrayGrammar(Grammar):
    @property
    def items(self):
        return [GExpr('array'), GExpr(['typename', 'typeexp', 'name', 'bits', 'bit', 'integer', 'real']), GExpr('bracketfunc')]

    def create_item(self, tokens, items, consumed):
        tokens.append(Expression(None, 'arraydef'))
        tokens[-1].children = [items[1], items[2]]


class GenericGrammarProxy(object):
    ''' Proxy object to wrap already instantiated grammars '''
    def __init__(self, grammar):
        self.grammar = grammar

    def __call__(self):
        return self.grammar


ConcatExpres = [Expressions.Name, Expressions.String, Expressions.Bitselect, Expressions.OpConcat, Expressions.Funcall]
_ConcatGrammar = BinaryOpGrammar(Expressions.OpConcat, [
    GExpr(ConcatExpres), GToken(Tokens.Colon), GExpr(ConcatExpres)
])
ConcatGrammar = GenericGrammarProxy(_ConcatGrammar)
_registered_grammars['stage3'].append((0, ConcatGrammar()))

_AssignGrammar = BinaryOpGrammar('assign', [
    GExpr(['name', 'real', 'bitselect', 'tuple', 'paren', 'bracketfunc', 'typed_variable', 'destructure', 'substruct', 'load']), GExpr('assignment'), GExpr()
])
AssignGrammar = GenericGrammarProxy(_AssignGrammar)
_registered_grammars['stage3'].append((1, AssignGrammar()))


NumExp = GExpr([Expressions.Name, Expressions.Number, Expressions.Funcall, Expressions.Paren, 
                Expressions.Bracket, Expressions.BoolNot, Expressions.Negation, 'op_*', 'real',
                Expressions.String, Expressions.Curly, Expressions.Load, 'bracketfunc', Expressions.Bitselect]
)
_NumericGrammars = [BinaryOpGrammar(name, [NumExp, op, NumExp])  
                   for op, name in [(GToken(Tokens.Star),  'op_num_mult'),
                                    (GExpr('DIV'),         'op_num_div'),
                                    (GExpr('MOD'),         'op_num_mod'),
                                    (GExpr('bsr'),         'op_num_bsr'),
                                    (GExpr('bsl'),         'op_num_bsl'),
                                    (GToken(Tokens.Caret), 'op_num_pow'),
                                    (GToken(Tokens.Plus),  'op_num_add'),
                                    (GToken(Tokens.Minus), 'op_num_sub')]
]
NumericGrammars = GenericGrammarProxy(_NumericGrammars)
_registered_grammars['numeric'] = list(enumerate(NumericGrammars()))


_LogicalGrammars = [BinaryOpGrammar(name, [NumExp, op, NumExp])  
                    for op, name in [(GToken(Tokens.LessThan), 'op_less_than'),
                                     (GToken(Tokens.GreaterThan), 'op_greater_than'),
                                     (GExpr('gteq'), 'op_gteq'),
                                     (GExpr('leq'), 'op_leq'),
                                     (GExpr('eq'), 'op_equal'),
                                     (GExpr('neq'), 'op_not_equal'),
                                     (GExpr('land'), 'op_logical_and'),
                                     (GExpr('lor'), 'op_logical_or'),
                                     (GExpr('IN'), 'op_in')]
]
LogicalGrammars = GenericGrammarProxy(_LogicalGrammars)
_registered_grammars['logical'] = list(enumerate(LogicalGrammars()))


_RealGrammar = BinaryOpGrammar('real', [
    GExpr(Expressions.Number), GToken(Tokens.Dot), GExpr(Expressions.Number)
])
RealGrammar = GenericGrammarProxy(_RealGrammar)
_registered_grammars['stage1'].append((7, RealGrammar()))


_LoadGrammar = BinaryOpGrammar('load', [
    NumExp, GToken(Tokens.Dot), NumExp
])
LoadGrammar = GenericGrammarProxy(_LoadGrammar)
_registered_grammars['stage2'].append((15, LoadGrammar()))


# Sort each grammar stage according to each grammars sort key
for kk in _registered_grammars.keys():
    _registered_grammars[kk] = sorted(_registered_grammars[kk], key=lambda (x, _): x)
