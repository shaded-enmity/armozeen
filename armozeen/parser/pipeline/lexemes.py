from armozeen.types import BuiltinTypes, Keywords, Tokens
from armozeen.parser.pipeline import lexer

_Parens = lexer.TokenPairCollapser('(', ')', ';', 'paren')
_Curlys = lexer.TokenPairCollapser('{', '}', '', 'curly')
_Brackets = lexer.TokenPairCollapser('[', ']', '', 'bracket')
_DoubleQuotedString = lexer.Stringifier('"', type_='sstring')
_SingleQuotedString = lexer.Stringifier("'")
_CommentRemover = lexer.CommentRemover()
_WhitespaceRemover = lexer.WhitespaceRemover()
_PairWiseCollapser = lexer.PairwiseCollapser([
    (Tokens.Plus,        Tokens.Equals,      'add_assign'),
    (Tokens.Minus,       Tokens.Equals,      'sub_assign'),
    (Tokens.Star,        Tokens.Equals,      'mul_assign'),
    (Tokens.Backslash,   Tokens.Equals,      'div_assign'),
    (Tokens.GreaterThan, Tokens.Equals,      'gteq'),
    (Tokens.LessThan,    Tokens.Equals,      'leq'),
    (Tokens.Equals,      Tokens.Equals,      'eq'),
    (Tokens.Exclamation, Tokens.Equals,      'neq'),
    (Tokens.GreaterThan, Tokens.GreaterThan, 'bsr'),
    (Tokens.LessThan,    Tokens.LessThan,    'bsl'),
    (Tokens.Or,          Tokens.Or,          'lor'),
    (Tokens.And,         Tokens.And,         'land'),
    (Tokens.Dot,         Tokens.Dot,         'range'),
    (Tokens.Backslash,   Tokens.Backslash,   'comment'),
    (Tokens.Backslash,   Tokens.Star,        'ml_comment_start'),
    (Tokens.Star,        Tokens.Backslash,   'ml_comment_end')
])
_ConsecutiveWhitespace = lexer.Consecutive(Tokens.Space, 'indented')
_ConsecutiveTabs = lexer.Consecutive(Tokens.Tab, 'tab_indented')


_Keywords = lexer.Keywordizer(Keywords)
_Names = lexer.Nameizer()
_Bitselect = lexer.AngledExpressions()
#_Builtins = lexer.BuiltinTyper(BuiltinTypes)

_Unary = lexer.FindUnaryOperators()
_Tuples = lexer.FindTuples()
_Types = lexer.FindTypes()
_BitsType = lexer.HandleBitsType()
_Assignments = lexer.FindAssignment()
_FunctionDefs = lexer.FindFunctionDefinitions()
_FunctionCalls = lexer.FindFunctionCalls()
_InlineConditionals = lexer.FindInlineConditionals()

_IndentationCollapser = lexer.IndentationCollapser()
_ParenNewlineRemover = lexer.ParenNewlineRemover()
_EarlyLoadResolver = lexer.NameLoads()

'''
Lexemes = [_DoubleQuotedString, _SingleQuotedString, _PairWiseCollapser, _CommentRemover, 
           _ConsecutiveWhitespace, _ConsecutiveTabs, _Keywords, _Builtins, _Names, _Parens, _Curlys,
           _Brackets, _Bitselect, _Unary, _BitsType, _Types, _Tuples, _Assignments, _EarlyLoadResolver,
           _FunctionDefs, _FunctionCalls, _InlineConditionals, _IndentationCollapser, _WhitespaceRemover,
           _ParenNewlineRemover]
'''
Lexemes = [_DoubleQuotedString, _SingleQuotedString, _PairWiseCollapser, _CommentRemover, 
           _ConsecutiveWhitespace, _ConsecutiveTabs, _Keywords, _Names, _Parens, _Curlys,
           _Brackets, _Bitselect, _Unary, _BitsType, _Types, _Tuples, _Assignments, _EarlyLoadResolver,
           _FunctionDefs, _FunctionCalls, _InlineConditionals, _IndentationCollapser, _WhitespaceRemover,
           _ParenNewlineRemover]
