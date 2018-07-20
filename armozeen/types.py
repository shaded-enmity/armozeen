from pprint import pformat


class LanguageStage(object):
    (PreGrammars, ExpressionTree, AST, Interpret) = range(4)


class Token(object):
    def __init__(self, ch, position=None):
        self.char = ch
        self.position = position

    def __repr__(self):
        return self.char


class ArmozeenException(Exception): pass


class Tokens(object):
    All = '()+;={}[]-*/\\%&|.,\n\':"<>!? \t^\0'
    (LeftParen, RightParen, Plus, Semicolon, Equals,
     LeftCurly, RightCurly, LeftSquare, RightSquare, Minus,
     Star, Backslash, Slash, Percent, And, Or, Dot, Comma,
     NewLine, Quote, Colon, DoubleQuote, LessThan, GreaterThan,
     Exclamation, Question, Space, Tab, Caret, Nil) = All


class Expression(object):
    def __init__(self, parent, typ):
        self.type = typ
        self.children = []
        self.parent = parent

    def __getitem__(self, i):
        return self.children[i]

    def __repr__(self):
        return '{}: '.format(self.type) + pformat(self.children)


class Keywords(object):
    All = ['if', 'then', 'else', 'case', 'of', 'when', 'for', 'to', 'constant',
           'DIV', 'MOD', 'EOR', 'AND', 'OR', 'otherwise', 'assert', 'IN', 'while',
           'return', 'type', 'is', 'enumeration', 'elsif', 'array', 'downto',
           'repeat', 'until', 'import', 'as']
    (If, Then, Else, Case, Of, When, For, To, Const, Div, Mod, Xor, And,
     Or, Otherwise, Assert, In, While, Return, Type, Is, Enum, Elsif,
     Array, DownTo, Repeat, Until, Import, As) = All


class Expressions(object):
    Indentation = ["indented", "tab_indented"]
    (Indented, TabIndented) = Indentation
    Fundef = 'fundef'
    Funcall = 'funcall'
    Paren = 'paren'
    String = 'string'
    Root = 'root'
    TypeName = 'typename'
    Number = 'number'
    Name = 'name'
    Division = 'DIV'
    Modulo = 'MOD'
    XOR = 'EOR'
    Bits = 'bits'
    Type = 'type'
    Of = 'of'
    Is = 'is'
    If = 'if'
    IfThen = 'ifthen'
    Then = 'then'
    Else = 'else'
    Elsif = 'elsif'
    Tuple = 'tuple'
    Bracket = 'bracket'
    Enum = 'enum'
    Assignment = 'assignment'
    Bitselect = 'bitselect'
    Substruct = 'substruct'
    Comment = 'comment'
    MultiLineStart = 'ml_comment_start'
    MultiLineEnd = 'ml_comment_end'
    Typedef = 'typedef'
    Negation = 'negate'
    BoolNot = 'bnegate'
    Condition = 'condition'
    Branch = 'branch'
    InlineIf = 'inline_if'
    Block = 'block'
    TypeExp = 'typeexp'
    TypeVar = 'typevar'
    Curly = 'curly'
    Enumeration = 'enumeration'
    Case = 'case'
    When = 'when'
    Othwerise = 'otherwise'
    Return = 'return'
    Otherwise = 'otherwise'
    Parameters = 'parameters'
    Parameter = 'parameter'
    Load = 'load'
    TypedVariable = 'typed_variable'
    SizeExpression = 'size_expression'

    IfStatement = 'if_statement'
    OpConcat = 'op_concat'


class BuiltinTypes(object):
    All = ['bit', 'integer', 'boolean', 'real']
    (Bit, Integer, Boolean, Real) = All


class UnaryOp(Expression):
    def __init__(self, typ, item):
        super(UnaryOp, self).__init__(None, typ)
        self.children.append(item)


class TypeDef(Expression):
    def __init__(self, name, defn):
        super(TypeDef, self).__init__(None, Expressions.Typedef)
        self.children = [name, defn]


class Block(Expression):
    def __init__(self, level, parent, children=None):
        self.level = level
        super(Block, self).__init__(parent, Expressions.Block)
        self.children = children or []


class Type(Expression):
    def __init__(self, name, size=0, const=False, parent=None):
        super(Type, self).__init__(parent, Expressions.TypeExp)
        self.children = [name, size, const]

    def __repr__(self):
        return '{}<\n    {}\n, c={}>'.format(self.name, self.size, self.const)

    name = property(lambda self: self.children[0])
    size = property(lambda self: self.children[1])
    const = property(lambda self: self.children[2])


#class TypedVariable(Expression):
#	def __init__(self, type_, name, is_reference=False):
#		self.type = type_
#		self.name = name
#		self.is_reference = is_reference
#		super(TypedVariable, self).__init__(None, 'typed_variable')

class TypedVariable(Expression):
    def __init__(self, type_, name, is_reference=False):
        super(TypedVariable, self).__init__(None, 'typed_variable')
        self.children = [type_, None, name, is_reference]

    var_type = property(lambda self: self.children[0])
    name = property(lambda self: self.children[2])
    is_reference = property(lambda self: self.children[3])