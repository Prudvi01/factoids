def num_of_revi(articleName):
    document_text = open(articleName, 'r')
    text_string = document_text.read().lower()
    text_list = text_string.split()
    count = 0
    words = 0
    for word in text_list:
        if word == '<revision>':
            count +=1
        words += 1
    return count

def clean(text):
    text = text.replace('\t', ' ')
    text = spaces.sub(' ', text)
    text = dots.sub('...', text)
    text = re.sub(' (,:\.\)\]»)', r'\1', text)
    text = re.sub('(\[\(«) ', r'\1', text)
    text = re.sub(r'\n\W+?\n', '\n', text, flags=re.U)  # lines with only punctuations
    text = text.replace(',,', ',').replace(',.', '.')
    if options.keep_tables:
        # the following regular expressions are used to remove the wikiml chartacters around table strucutures
        # yet keep the content. The order here is imporant so we remove certain markup like {| and then
        # then the future html attributes such as 'style'. Finally we drop the remaining '|-' that delimits cells.
        text = re.sub(r'!(?:\s)?style=\"[a-z]+:(?:\d+)%;\"', r'', text)
        text = re.sub(r'!(?:\s)?style="[a-z]+:(?:\d+)%;[a-z]+:(?:#)?(?:[0-9a-z]+)?"', r'', text)
        text = text.replace('|-', '')
        text = text.replace('|', '')
    if options.toHTML:
        text = html.escape(text)
    return text