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