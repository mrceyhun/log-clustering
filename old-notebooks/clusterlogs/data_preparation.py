import re

class Regex(object):

    def __init__(self, messages):
        self.messages = messages
        self.messages_cleaned = None

    def process(self):
        """
        :return:
        """
        self.messages_cleaned = [0] * len(self.messages)
        for idx, item in enumerate(self.messages):
            #item = re.sub(r'(/[\w\./]*[\s]?)', ' ', item)
            #item = re.sub(r'([a-zA-Z0-9]+[_]+[\S]+)', ' ', item)
            #item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', ' ', item)
            #item = re.sub(r'[^\w\s]', ' ', item)
            #item = re.sub(' +', ' ', item)
            self.messages_cleaned[idx] =  item
        return self.messages_cleaned
