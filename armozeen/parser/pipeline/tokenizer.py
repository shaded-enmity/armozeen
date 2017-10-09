from armozeen.parser.pipeline import PipelineStage
from armozeen.types import Token


class Tokenizer(PipelineStage):
    def __init__(self, tokens):
        self.tokens = tokens

    def run(self, string):
        wordbuf, tokens = '', []
        linec, charc = 0, 0
        for c in string:
            if c in self.tokens:
                if wordbuf:
                    tokens.append(Token(wordbuf, (linec, (charc-len(wordbuf),charc))))
                    wordbuf = ''
                tokens.append(Token(c, (linec, charc)))
            else:
                wordbuf += c

            if c == '\n':
                linec += 1
                charc = -1

            charc += 1
        if wordbuf:
            tokens.append(Token(wordbuf, (linec, charc)))
        return tokens

