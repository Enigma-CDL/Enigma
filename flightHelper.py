# Definitions required to support dependant Lagrange settings (see further below)
# Base Weight calculation


HomeBaseWeightOffset =  4

def makeBaseWeight(id):
    # Weight is calculated as: 10 ** ( id + offset +1 )
    #
    return( 10 ** ( id + HomeBaseWeightOffset + 1))


# Todo: Rename a "start" to a "break". 

class Start:

    def __init__(self, id, lab):
        self.id = id
        self.lab = lab

class Node:

    def isStart(self):
        return isinstance(self.obj,Start)
    
    def isSegment(self):
        return isinstance(self.obj,Segment)
    
    def __init__(self, obj ):
        if ( isinstance(obj,Segment)):
            self.id = obj.id
            self.lab = obj.lab
            self.obj = obj
            #print("Node %d is %s" % (self.id, self.lab))
        elif ( isinstance(obj,Start)):
            self.id = obj.id
            self.lab = obj.lab
            self.obj = obj
            #print("We have Start State %d" % (self.id))
        else:
            raise Exception("Error on type. Got %s" % (type(obj)))



# Home Base definitions
#
# Each valid home base is defined to be assigned its own Weight
# Weight is added on the first node and subtracted on the last node. If bases match, net 0 effect, no penalty
#
#
# Variations: Try LCA alone, ATH alone or both LCA and ATH
#
# 

# Segment Class

class Segment:

    def getUarrtime(self):
        return ((self.getUdeptime() + self.ft))
                
    def getUdeptime(self):
        return ((self.depday - 1) * 1440 + self.deptime)
    
    def setT(self, T):
        self.T = T
        self.UT1 = self.getUdeptime()
        self.UT2 = T - ( self.getUdeptime() + self.ft)
        #print( self.UT1, self.UT2, T, self.getUdeptime, self.getUarrtime)
        
    def getUT(self):
        return( self.T - self.ft)

    def getUT1(self):
        return( self.UT1)

    def getUT2(self):
        return( self.UT2)
    
    def setCI(self,ci):
        self.ci = ci
        
    def setCO(self,co):
        self.co = co
        
    def getCI(self):
        return self.ci
        
    def getCO(self):
        return self.co
        
    def __init__(self, id, lab, dep, arr, deptime, arrtime, depday, arrday, HomeBases):
        self.id = id
        self.lab = lab
        self.dep = dep
        self.arr = arr
        self.deptime = deptime
        self.arrtime = arrtime
        self.depday = depday
        self.arrday = arrday
        self.ft = (arrday-1) * 1440 + arrtime - ( (depday-1) * 1440 + deptime )
        self.UT1 = 0
        self.UT2 = 0
        self.T = 0
        self.ci = 0
        self.co = 0
        
        # Establish the weights associated with departure and arrival airport
        # 
        # Only apply a weight for base airports
        #
        
        self.DepBaseWgt = 0 #NotHomeBaseWeight
        self.ArrBaseWgt = 0 #NotHomeBaseWeight
        
        if ( self.dep in HomeBases):
            self.DepBaseWgt = makeBaseWeight(HomeBases[dep])
            
        if ( self.arr in HomeBases):
            self.ArrBaseWgt = makeBaseWeight(HomeBases[arr])
        
        #  Not required ... yet. The idea of carrying forward a weight from node to node may still be useful
        # self.TransitionBaseWeight = self.DepBaseWgt - self.ArrBaseWgt

# Create Start objects for adding as special edges


# print("Testing Node class")
# n1 = Node(Segment(1, "101", "LCA", "ATH", 600, 695, 1, 1))
# n2 = Node(State(N*N,"start"))
# print( n1.isStart() )
# print( n2.isStart() )
# n3 = Node(10)    # raises exception as expected
        

# Transition Weight
#
#
# TODO: Rethink the Segment and State classes.
#       We need to have Segments and States coexist in the edges.
#       Therefore we need a Vertex class that can be either a Segment or a Start State
# 

class TransitionWeight:
    
    def __init__(self,node1,node2):
        #
        # Detect the case we have:
        #
        # start-node
        # node-start
        # start-start
        #
        self.gap = 0
        if ( node1.isSegment()):   
            if ( node2.isStart()):     # C -> S
                self.TransitionBaseWeight = 0 # node1.obj.TransitionBaseWeight - node1.obj.ArrBaseWgt
            else:                      # C -> C : Standard Weight Transition
                self.TransitionBaseWeight = ConnectWeight(node1.obj,node2.obj).TransitionBaseWeight
        else:
            if ( node2.isSegment()):   # S -> C
                self.TransitionBaseWeight = 0 # node2.obj.TransitionBaseWeight - node2.obj.ArrBaseWgt
            else:                      # S -> S
                self.TransitionBaseWeight = 0
                
    
    
    
# Connection Weight
#
# Includes: Time gap, origin
#

class ConnectWeight:

    # 1) Connection Time gap. If negative make it zero. Zero gaps will be excluded from edges.  
    # 2) HomeBase switch weight
    # 3) Matching Airport connection: If does not match, assign NegativeWeight
    #

    def __init__ (self, seg1, seg2):
        
        #NegativeWeight = 10**(HomeBaseWeightOffset+1)  # TODO: Parametrize this definition

        # Include FT of nodes to allow for constraints on maximum flight time
        #
        
        self.ft = seg1.ft + seg2.ft
        
        # Include the time gap for the connection

        self.gap = gap(seg1,seg2)
        
        if ( self.gap < 0 ): 
            self.gap = 0 # abs(NegativeWeight + self.gap)

        # Connect Weight for Airports:
        #
        # bi,j: is the difference between the seg2.arr weight and seg1.dep weight
        # This is the TransitionalBaseWeight for the connection.
        #
        # TODO: Decide on abs value or signed
                    
        self.TransitionBaseWeight = seg2.ArrBaseWgt - seg1.DepBaseWgt
        
        # TODO: Decide how to deal with a gap when not connecting at the same airport
        
        if ( seg2.dep != seg1.arr ):
            self.gap = 0 # NegativeWeight;
        #print(self.__dict__)
        
def gap(seg1,seg2):
	return( seg2.depday * 1440 + seg2.deptime - (seg1.depday * 1440 + seg1.deptime + seg1.ft ))


# Constraints as functions so we can easily turn them on or off

class TripGen:
    
    def __init__(self,N,segments):
        self.N = N
        self.segments = segments
        
    # New test for constraint1- Works much better. Note how the coef are updated: += 
    def const_quad_nodes3(self,Name, Q, LG, coef_lin,coef_quad,coef_const):
        N = self.N
        count_lin = 0;
        count_quad = 0;
        for row in range(N):
            for u in range(N):
                indx = row * N + u
                Q[(indx,indx)] += (LG * coef_lin)
                for v in range(u+1,N):
                    jndx = row * N + v
                    Q[(indx,jndx)] +=  coef_quad * LG
                    count_quad+=1
        print(Name,LG, coef_lin,coef_quad,coef_const)
        print("Acted on : %d lins, %d quads" % (count_lin, count_quad))
        #print(Q)


     # Revised and Corrected
    # Coefficient handling corrected
    def const_quad_rows(self,Name, Q, LG, coef_lin,coef_quad,coef_const):
        N = self.N
        count_lin = 0;
        count_quad = 0;
        for row in range(N):
            origin = row * N
            for node in range(N):
                indx = origin + node
                Q[(indx,indx)] += (LG * coef_lin)
                count_lin+=1
                for row2 in range( row + 1, N):
                    jndx = row2 * N + node
                    Q[(indx,jndx)] +=  coef_quad * LG
                    count_quad+=1
        print(Name,LG, coef_lin,coef_quad,coef_const)
        print("Acted on : %d lins, %d quads" % (count_lin, count_quad))
    #print(Q)

    # Change: Dec Lin, Inc Quad for constraints
    def const_quad_states(self,Name, Q, LG, coef_lin, coef_quad, coef_const):
        N = self.N
        for row in range(N):
            rndx = N**2 + row  # N bits after the main nodes matrix
            Q[(rndx,rndx)] += (LG * coef_lin)
            for j in range(row+1, N):
                jndx = N**2 + j
                Q[(rndx,jndx)] +=  coef_quad * LG
        print(Name,LG, coef_lin,coef_quad,coef_const)
        #print(Q)

    # WIP: Reworking the loop. Need to consider consecutive rows.

    def const_edges_connect(self,Name, Q, LG, G):
        N = self.N
        segments = self.segments
        for row in range(N-1):
            row2 = row + 1
            for node in range(N):
                for node2 in range(N):
                    indx = row * N + node
                    jndx = row2 * N + node2

                    if not G.has_edge(segments[node],segments[node2]):

                        # Penalize the lack of edge
                        #print("Penalizing %d to %d, connection not allowed" % (node,node2))
                        Q[(indx,jndx)] += LG

        print(Name)
        #print(Q)

    # ================================================================
    # Return to base : penalizing on start and depenalizing on return
    # 
    # const_location_start : Will add the base weight of the airport
    # const_location_return: Will subtract base weight of the airport
    # 
    # a Net Zero means we have a cycle returning to the start airport
    #
    # ================================================================
    #    for n1,n2,data in G.edges(data=True):

    def const_location_start(self,Name, Q, LG, G):
        N = self.N
        segments = self.segments

        # for edges connecting a start to a segment install the base weight
        for s1,v1,cw in G.edges(data=True):

            for r in range(N):
                s = N*N + r
                for v in range(N):
                    sndx = s
                    vndx = r * N + v
                    if ( s1.id == s ) and (v1.id == segments[v].id):
                        Q[(sndx,vndx)] -= LG * segments[v].obj.DepBaseWgt
        print(Name)
        #print(Q)

    def const_location_return(self,Name, Q, LG, G):
        N = self.N
        segments = self.segments

        # for edges connecting a segment to a start retract the base weight
        for v1,s1,cw in G.edges(data=True):

            for r in range(2,N):
                s = N*N + r
                for v in range(N):
                    sndx = r
                    vndx = (r-1) * N + v  # Previous row node
                    if ( s1.id == s ) and (v1.id == segments[v].id):
                        Q[(vndx,sndx)] += LG * segments[v].obj.ArrBaseWgt

        print(Name)
        #print(Q)

        
    # Objective for minimizing non-flying time
    # Coefficients calculated for each segment
    # We use 
    # (Segment.getUT() ** 2) * coef_lin 
    # (Segment.getUT() * 2 * coef_quad
    # We do not apply a Lagrange for objectives
    #

    def objective_quad_nodes(self, Name, Q, coef_lin,coef_quad,coef_const):
        N = self.N
        segments = self.segments
        count_lin = 0;
        count_quad = 0;
        for row in range(N):
            for u in range(N):
                undx = row * N + u
                count_lin+=1
                Q[(undx,undx)] += (segments[u].obj.getUT() ** 2 * coef_lin) # Base line unallocated time
                #print("a%d" %(u)," = ", (segments[u].obj.getUT() ** 2 * coef_lin))
                for v in range(u+1,N):
                    vndx = row * N + v
                    # Base line unallocated time 
                    Q[(undx,vndx)] +=  coef_quad * segments[v].obj.getUT() 
                    #print("b%d,%d" % (u,v), " = ", coef_quad * segments[v].obj.getUT() )
                    # Minus pair contribution : - ( 1.UT2 + 2.UT1 ) time gap remains
                    Q[(undx,vndx)] += -  (segments[u].obj.getUT2()+segments[v].obj.getUT1()) # Removed coef_quad *
                    #print("b%d,%d" % (u,v), " plus ", -  (segments[u].obj.getUT2()+segments[v].obj.getUT1() ))
                    count_quad+=1
        print(Name,1, coef_lin,coef_quad,coef_const)
        print("Objective Acted on : %d lins, %d quads" % (count_lin, count_quad))
        #print(Q)
    # Objective for cancelling the gap when a cycle starts and replacing it with CheckIn time (1 to N+1)
    # or cancelling  the reaminder of unallocated time when a cycle ends (2 to N )

    def objective_quad_states(self, Name, Q, coef_lin, coef_quad, coef_const, G):

        for node1, node2 in G.edges(data=False):
            N = self.N
            segments = self.segments
            
            # Process start to segment (new cycle)

            if ( node1.isStart() and node2.isSegment()):

                for row in range(N):

                    sndx = N*N + row               # Start state per row
                    undx = row * N + node2.id - 1  # Node starting or continuing

                    # Case S to C : node2 is a start of a cycle
                    # Cancel UT1 and replace it with CheckIn time
                    print("Node %d start, cancelling %d, adding %d" % ( node2.id, node2.obj.getUT1(), node2.obj.getCI()))
                    Q[(sndx,undx)] +=  coef_quad * ( node2.obj.getCI() - node2.obj.getUT1())

            # Process segment to start (end of a previous cycle)

            if ( node1.isSegment() and node2.isStart()):

                # We start at row 1 and refer to the previous row
                for row in range(1,N):

                    undx = (row - 1) * N + node1.id - 1  # Node ending a cycle
                    sndx = N*N + row                     # Start state per row

                    # Case C to S : node1 is the end of a cycle
                    # Cancel UT2 and replce it with CheckOut time
                    print("Node %d end, cancelling %d, adding %d" % ( node1.id, node1.obj.getUT2(), node1.obj.getCO()))
                    Q[(undx,sndx)] +=  coef_quad * ( node1.obj.getCO() - node1.obj.getUT2())

        print(Name,coef_lin,coef_quad,coef_const)
        #print(Q)
        
    def print_trip(self,result):
        variables = result[0]
        energy = result[1]
        N = self.N
        segments = self.segments
        
        print("Energy %f" % (energy))
        sndx = N*N
        for row in range(N):
            origin = row * N
            state = 0
        
            if ( sndx+row < len(variables)):
                state = variables[sndx+row]

            for node in range(N):
                n = origin + node

                if ( variables[n] == 1):
                    if ( state == 1):
                        print("Start")
                    print(row, segments[node].obj.id, segments[node].obj.lab, segments[node].obj.dep, segments[node].obj.arr)
        print("---------------")

    # print(len(sampleset._record), sampleset._record[0])

    def print_all(self, sampleset, max = 3):
        for res in sampleset.data():
            self.print_trip(res)
            max = max - 1
            if ( max <= 0 ): break


