import pandas as pd


def readGslibPoints(filePath):
    columnNames = []
    with open(filePath, 'r') as fi:
        fi.readline()
        n = int(fi.readline().strip())
        for i in range(n):
            columnNames.append(fi.readline().strip())
        fi.close()
    df = pd.read_csv(filePath, sep=',|\s+', skiprows=n + 2, header=None)
    df.columns = columnNames
    return df


def saveGslibPoints(filePath, colNames, ndarray, title="#"):
    fo = open(filePath, 'w')
    fo.write(title + '\n')
    fo.write('{}\n'.format(len(colNames)))
    for name in colNames:
        fo.write(name + '\n')
    df = pd.DataFrame(ndarray)
    df.to_csv(fo, sep=' ', line_terminator='\n', index=False, header=False)
    fo.close()


def saveGslibGrid(filePath, dim, begin, step, colNames, ndarray, comment=None):
    fo = open(filePath, 'w')
    if comment is not None:
        fo.write('#' + comment + '\n')
    line = ' '.join(list(map(lambda x: str(x), dim)))
    fo.write(line + '\n')
    line = ' '.join(list(map(lambda x: str(x), begin)))
    fo.write(line + '\n')
    line = ' '.join(list(map(lambda x: str(x), step)))
    fo.write(line + '\n')
    df = pd.DataFrame(ndarray)
    df.columns = colNames
    df.to_csv(fo, sep=' ', line_terminator='\n', index=False)
    fo.close()
