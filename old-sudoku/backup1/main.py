import init
import utils

def format (listOfLists):
    for i in range (0, init.size):
        for j in range (0, init.size):
            print listOfLists[i][j],
        print "i=",i


# variables are in: init.horiz, init.vert, init.block
# create a data structure which maps coordinate pairs to possible values.

possible=[[set([str(x) for x in range(1,init.size+1)]) for x in range(0, init.size)] for x in range(0, init.size)]

# easier to check each 3x3 block. 
# first round of elimination takes out all the obvious 

for x in range (0, init.size):
    # before doing anything else remove all the items
    # from the possible vector which are already in this block
    inThisBlock = set(init.block[x])
    for y in range (0, init.size):
        # iterate over each block.
        
        item = init.block[x][y]
        i,j = utils.blockToIndex (x,y)
        if item == '.':
            # we don't know what goes here
            possible[i][j] = utils.setDiff (possible[i][j], inThisBlock)
            possible[i][j] = utils.setDiff (possible[i][j], set(init.horiz[i]))
            possible[i][j] = utils.setDiff (possible[i][j], set(init.vert[j]))

            if len(possible[i][j]) == 1:
                item = list(possible[i][j])[0]
                init.horiz[i][j] = item
                init.vert[j][i] = item
                init.block[x][y] = item
                possible[i][j] = set([])
                
        else:
            possible[i][j] = set([])
print "After initial pass:"
format (init.horiz)

print "Beginning more passes"

t = 0

# keep doing this until there are no more holes
while not utils.isDone (possible):

    counter = 0 # use this to keep track of how many things changed in elimination
    for i in range (0, init.size):
        for j in range (0, init.size):
            x,y = utils.indexToBlock (i,j)

            # First, eliminate anything that is already placed            
            possible[i][j] = possible[i][j] - set(init.horiz[i])
            possible[i][j] = possible[i][j] - set(init.vert[j])
            possible[i][j] = possible[i][j] - set(init.block[x])
            #print "[",i,"][",j,"] After filtering, possible values are ",possible[i][j]    

            # if we have eliminated everything except one item,
            # that is the correct item
            if len(possible[i][j]) == 1:
                item = list(possible[i][j])[0]
                init.horiz[i][j] = item
                init.vert[j][i] = item
                init.block[x][y] = item
                possible[i][j] = set([])
                counter = counter+1
                print "Set item[",i,"][",j,"] to ",item," block is ",init.block[x]


            # The next check is to see if an item is the only one that can
            # fit in a cell. Do this by brute force: check if it can go into
            # any other cell in that block. 
            # only do the next check if a hole has not been filled
            if init.horiz[i][j] == '.':
                lp = list(possible[i][j])
                for p in range (0, len(lp)):
                    item = lp[p]
                    #print "Checking item ",item," against cell[",i,"][",j,"]"
                    if init.isOnlyCell (i,j,item):
                        # now we want to remove it and break
                        init.horiz[i][j] = item
                        init.vert[j][i] = item
                        init.block[x][y] = item
                        possible[i][j] = set([])
                        counter = counter+1
                        #print "Elimination: Set item[",i,"][",j,"] to ",item," row is ",init.horiz[i]," column is ", init.vert[j]," block is ",init.block[x]
                        break
    print "After elimination pass ",t
    #format (possible)
    format (init.horiz)


    # the last test is to look at all the possible values in each row
    if counter == 0:
        for i in range (0, init.size):
            # will need to iterate a few times over the same row. 
            union = set([])
            for j in range (0, init.size):
                # first, get the set that is the union of all the possible values
                # for this row
                union = union | possible[i][j]
            unionlist = list(union)
            dictionary = dict([(unionlist[t],[]) for t in range(0, len(unionlist))])
            # keys of the dictionaries: possible items
            # values: cells that can have the key as a possible value
            for j in range (0, init.size):
                lp = list (possible[i][j])
                # Iterate over all of the possible values.
                # build up the dictionary
                for k in range (0, len (lp)):
                    val = dictionary[lp[k]]
                    val.append (j)
                    
            for j in range (0, init.size):
                lp = list (possible[i][j])
                for k in range (0, len(lp)):
                    # look up the value from the dictionary
                    otherIndexes = dictionary[lp[k]]
                    if len(otherIndexes) == 1:
                        print "Item ",lp[k]," present only in possible[",i,"][",j,"]"
                
                        x,y = utils.indexToBlock (i,j)
                        item = lp[k]
                        if (possible[i][j] != [] and
                            item not in set(init.horiz[i]) and
                            item not in set(init.vert[j]) and
                            item not in set(init.block[x])):
                        
                            init.horiz[i][j] = item
                            init.vert[j][i] = item
                            init.block[x][y] = item
                            possible[i][j] = set([])
                            print "Pass ",t," guess: Set item[",i,"][",j,"] to ",item," row is ",init.horiz[i]," column is ", init.vert[j]," block is ",init.block[x]
                            break;
                        else: print "Pass ",t," guess: CANNOT SET item[",i,"][",j,"] to ",item," row is ",init.horiz[i]," column is ", init.vert[j]," block is ",init.block[x]," possible is ",possible[i][j]
                    
    
        print "After guess pass ",t
        #format (possible)

        format (init.horiz)
    t = t+1

print "Done after ",t-1," passes"
print "Verified: ",init.verify ()
#format (possible)
#format (init.horiz)


            
            
            



