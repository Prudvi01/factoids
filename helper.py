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

def findPosper(x, resultlength):
    posper = 0
    for i in x:
        if i > 0:
            posper += 1
    return (posper/resultlength) * 100

def justsent(temp, limit = 90000):
    x = 0
    newsent = []
    for i in temp:
        sent = i - x
        if sent <= limit and sent >= - limit:
            newsent.append(sent)
        x = i
    return newsent

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out