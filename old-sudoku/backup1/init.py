import utils
import sys

inputfile = sys.argv[1]
input = open(inputfile, 'r')

#input=open('./input.txt','r')
#input=open('./input2.txt','r')
#store the lines in a list of lists
size=9

control=set([str(x) for x in range(1,size+1)])

horiz=[[] for x in range(0, size)]
for i, line in enumerate (input):
    lineClean = line.strip()
    for char in lineClean[:]:
        horiz[i].append (char);

#print horiz
# now have all of the lines.
# start solving
# first build data structures to keep track of which fields have holes

# initialise for vertical lines.
# also blocks of 3
vert=[[0 for x in range (0, size)] for x in range(0, size)]
#print "vert=",vert
block=[[0 for x in range (0, size)] for x in range(0, size)]
#print "block=",block

for i in range (0, size):
    for j in range(0,size):
#        print "vert(",j,",",i,")=horiz(",i,",",j,")"
        vert[j][i] = horiz[i][j]
        indexOfBlock, indexInBlock = utils.indexToBlock (i,j)
        #indexOfBlock = (i/3)*3 + j/3
        #indexInBlock = (i%3)*3 + j%3
        # 0  1  2  == 0   1   2
        # 9  10 11 == 3+0 3+1 3+2
        # 18 19 20 == 6+0 6+1 6+2
        #print "i=",i,",j=",j,",block[",indexOfBlock,"][",indexInBlock,"]=",horiz[i][j]
        block[indexOfBlock][indexInBlock] = horiz[i][j]



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
            #print "[",x,"][",t,"] maps to [",i2,"][",j2,"]"
            if not ((i2 == i) and (j2  == j)): 
                #print "Checking item ",item," against [",i2,"][",j2,"]"
                #print "Checking item ",item," against horiz[",i2,"]=",horiz[i2]
                value = value and item in horiz[i2]
                if value == False:
                    #print "Checking item ",item," against vert[",j2,"]=",vert[j2]
                    value = item in vert[j2]
                if value == False: return value
    print "item ",item," is the only choice for [",i,"][",j,"]: ",value
    return value

#9 1 5 6 7 4 2 3 8 i= 0
#8 2 7 1 5 3 4 9 6 i= 1
#6 . 3 8 1 9 2 5 7 i= 2
#. . . . . 6 . 8 4 i= 3
#. 9 4 7 . . 5 2 . i= 4
#. 8 6 4 . 5 . . . i= 5
#1 7 . 5 . 8 3 4 6 i= 6
#7 4 9 3 6 1 8 . 5 i= 7
#. 6 8 9 4 2 1 7 . i= 8


print isOnlyCell (2, 6, '2')


def verify():
    value = True
    for i in range (0, size):
        value = value and (control - set(horiz[i]) == set([]))
        value = value and (control - set(vert[i]) == set([]))
        value = value and (control - set(block[i]) == set([]))
    return value


