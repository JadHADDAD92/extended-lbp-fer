""" module to save / load objects """

import sys
import bz2
import _pickle as cPickle

def save(filename, myobj):
    """
    save object to file using pickle
    
    @param filename: name of destination file
    @type filename: str
    @param myobj: object to save (has to be pickleable)
    @type myobj: obj
    """
    
    try:
        file = bz2.BZ2File(filename, 'wb')
    except IOError as details:
        sys.stderr.write('File ' + filename + ' cannot be written\n')
        sys.stderr.write(details)
        return
    
    cPickle.dump(myobj, file)
    file.close()



def load(filename):
    """
    Load from filename using pickle
    
    @param filename: name of file to load from
    @type filename: str
    """
    
    try:
        file = bz2.BZ2File(filename, 'rb')
    except IOError as details:
        sys.stderr.write('File ' + filename + ' cannot be read\n')
        sys.stderr.write(details)
        return False
    
    myobj = cPickle.load(file)
    file.close()
    return myobj
