import pandas as pd


def readGslibPoints(filePath):
    '''
    Read points data from gslib file.

    :param filePath: str.
    :return: DataFrame object.
    '''
    columnNames = []
    with open(filePath, 'r') as fi:
        fi.readline()
        n = int(fi.readline().strip())
        for i in range(n):
            columnNames.append(fi.readline().strip())
        fi.close()
    df = pd.read_csv(filePath, sep=',|\s+', skiprows=n + 2, header=None, engine="python")
    df.columns = columnNames
    return df


def saveGslibPoints(filePath, colNames, ndarray, comment="#"):
    '''
    Save points data to gslib file.

    :param filePath: str.
    :param colNames: list, column names.
    :param ndarray: ndarray, points and properties data.
    :param comment: str, comment on the first line of the gslib file.
    :return: None.
    '''
    fo = open(filePath, 'w')
    fo.write(comment + '\n')
    fo.write('{}\n'.format(len(colNames)))
    for name in colNames:
        fo.write(name + '\n')
    df = pd.DataFrame(ndarray)
    df.to_csv(fo, sep=' ', line_terminator='\n', index=False, header=False)
    fo.close()


def saveGslibGrid(filePath, dim, begin, step, colNames, ndarray, comment=None):
    '''
    Save rectilinear grid data to gslib file.

    :param filePath: str.
    :param dim: list, [xnum, ynum, znum], xnum indicates the number of x coordinates.
    :param begin: list, [x,y,z], begin coordinates.
    :param step: list, [xstep,ystep,zstep], xstep indicates the step on x axis.
    :param colNames: list, column names.
    :param ndarray: ndarray, points and properties data.
    :param comment: str, comment on the first line of the gslib file.
    :return: None.
    '''
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
