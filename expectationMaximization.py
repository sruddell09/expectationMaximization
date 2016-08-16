import logfunc
from math import log
import pickle

def process_line(line):
    label, bow = line.strip().split('\t')
    bow = bow.split()
    return label, bow

def e_step(bow, theta):
    logjoint = [log(0.5), log(0.5)]
    for w in bow:
        for c in theta:
            if w in theta[c]:
                logjoint[int(c)] += log(theta[c][w])
            else:
                logjoint[int(c)] += log(theta[c]['<UNK>'])

    return logfunc.posterior(logjoint)

def m_step(x, z):
    theta = {}
    for bow in range(len(x)):
        for c in range(0, 2):
            c = str(c)
            if not c in theta:
                theta[c] = {}
            theta[c]['<UNK>'] = 0.0
            for w in x[bow]:
                if not w in theta[c]:
                    theta[c][w] = 0.0
                theta[c][w] += z[bow][int(c)]
    for c in theta:
        for w in theta[c]:
            theta[c][w] += 0.1
    for c in theta:
        ts = sum(theta[c].values())
        for w in theta[c]:
            theta[c][w] /= ts

    return theta

if __name__ == '__main__':
    
    x = []

    f = open("sent.100")

    for line in f:
        label, bow = process_line(line)
        x.append(bow)
    f.close()

    theta = pickle.load(open('sent.theta.emotions'))

    n = 10

    for i in range(n - 1):
        z = []
        for bow in x:
            z.append(e_step(bow, theta))
        theta = m_step(x, z)

    score = 0.0

    f = open('sent.100')

    for line in f:
        label, bow = process_line(line)
        logjoint = e_step(bow, theta)
        if logjoint[0] > logjoint[1]:
            result = '0'
        else:
            result = '1'
        if result == label:
            score += 1
    f.close()

    print "Accuracy: %.0f%%" % (score)


