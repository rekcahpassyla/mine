import utils
import sys

inputfile = sys.argv[1]
input = open(inputfile, 'r')

#store the lines in a list of lists
size=9

control=set([str(x) for x in range(1,size+1)])


# variables are in: init.horiz, init.vert, init.block
# create a data structure which maps coordinate pairs to possible values.

possible=[[set([str(x) for x in range(1,size+1)]) for x in range(0, size)] for x in range(0, size)]


horiz=[[] for x in range(0, size)]
for i, line in enumerate (input):
    lineClean = line.strip()
    for char in lineClean[:]:
        horiz[i].append (char);


# first build data structures to keep track of which fields have holes

# initialise for vertical lines.
# also blocks of 3
vert=[[0 for x in range (0, size)] for x in range(0, size)]
#print "vert=",vert
block=[[0 for x in range (0, size)] for x in range(0, size)]
#print "block=",block

for i in range (0, size):
    for j in range(0,size):
        vert[j][i] = horiz[i][j]
        x,y = utils.indexToBlock (i,j)
        block[x][y] = horiz[i][j]



#print vert
#print block



# returns true if horiz[i][j] is the only place that this item can go
# we assume that this is a partially filled grid
# so we do not need to check for horizontal, vertical, block etc. 
def isOnlyCell (i, j, item):
    if horiz[i][j] != '.': return False
    x,y = utils.indexToBlock (i,j)
    # test all of the squares in the block
    # if this item was in any of the other squares,
    # would it invalidate?
    value = True
    # first of all, has to be blank
    #print "[",i,"][",j,"]: Checking item ",item," against ",horiz[i][j]
    value = value and (horiz[i][j] == '.')
    # must be not already in the horizontal
    #print "[",i,"][",j,"]Checking item ",item," against ",horiz[i]
    value = value and item not in horiz[i]
    # must be not already in the vertical
    #print "[",i,"][",j,"]Checking item ",item," against ",vert[j]
    value = value and item not in vert[j]
    # if the item can't be in any of the blanks,
    # this function must return true. 
    if value == False: return value
    for t in range (0, size):
        if block[x][t] == '.': # a blank
            i2, j2 = utils.blockToIndex (x,t)
            if not ((i2 == i) and (j2  == j)): 
                value = value and item in horiz[i2]
                if value == False:
                    value = item in vert[j2]
                if value == False: return value
    return value

def verify():
    value = True
    for i in range (0, size):
        value = value and (control - set(horiz[i]) == set([]))
        value = value and (control - set(vert[i]) == set([]))
        value = value and (control - set(block[i]) == set([]))
    return value



def setItem(i,j,item):
    horiz[i][j] = item
    vert[j][i] = item
    x,y = utils.indexToBlock (i,j)
    block[x][y] = item
    possible[i][j] = set([])
