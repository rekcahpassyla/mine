

def indexToBlock(i, j):
    indexOfBlock = (i/3)*3 + j/3
    indexInBlock = (i%3)*3 + j%3
    return  (indexOfBlock, indexInBlock)

def blockToIndex(blockIndex, indexInBlock):
    # blockIndex/3 gives initial vertical displacement: which multiple of 3
    # indexInBlock/3 gives additional displacement
    i = 3*(blockIndex/3) + indexInBlock/3
    # blockIndex%3 gives initial horizontal displacement
    # indexInBlock%3 gives additional displacement
    j = 3*(blockIndex%3) + indexInBlock%3
    return (i,j)

def setDiff (setA, setB):
    return setA - setB
    

# list will be a list of lists of sets    
def isDone (list):
    value = True
    for i in range (0, len(list)):
        for j in range (0, len(list[i])):
            value = value and (len(list[i][j])) == 0
            if value == False: return False
    return value
