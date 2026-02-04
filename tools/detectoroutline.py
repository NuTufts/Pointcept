import os,sys

class DetectorOutline:
    def __init__(self):
        self.tpc = [[0.0,256.0],
                    [-117.0,117.0],
                    [0.0,1036.0]]
        self.dettick_range = [0.0, 9600.0]
        self.tpctrig_tick = 3200.0
        self.detx_range = [ (self.dettick_range[0]-self.tpctrig_tick)*0.5*0.111,
                            (self.dettick_range[1]-self.tpctrig_tick)*0.5*0.111 ]

        self.top_pts  = [ [self.tpc[0][0],self.tpc[1][1], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][1], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][1], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][1], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][1], self.tpc[2][0]] ]
        self.bot_pts  = [ [self.tpc[0][0],self.tpc[1][0], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][0], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][0], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][0], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][0], self.tpc[2][0]] ]
        self.up_pts   = [ [self.tpc[0][0],self.tpc[1][0], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][0], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][1], self.tpc[2][0]],
                          [self.tpc[0][0],self.tpc[1][1], self.tpc[2][0]],
                          [self.tpc[0][0],self.tpc[1][0], self.tpc[2][0]] ]
        self.down_pts = [ [self.tpc[0][0],self.tpc[1][0], self.tpc[2][1]],
                          [self.tpc[0][1],self.tpc[1][0], self.tpc[2][1]],
                          [self.tpc[0][1],self.tpc[1][1], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][1], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][0], self.tpc[2][1]] ]

                
    def getlines(self,color=(255,255,255)):

        # top boundary
        Xe = []
        Ye = []
        Ze = []

        for boundary in [self.top_pts, self.bot_pts, self.up_pts, self.down_pts]:
            for ipt, pt in enumerate(boundary):
                Xe.append( pt[0] )
                Ye.append( pt[1] )
                Ze.append( pt[2] )
            Xe.append(None)
            Ye.append(None)
            Ze.append(None)
        
        
        # define the lines to be plotted
        lines = {
            "type": "scatter3d",
            "x": Xe,
            "y": Ye,
            "z": Ze,
            "mode": "lines",
            "name": "",
            "line": {"color": "rgb(%d,%d,%d)"%color, "width": 5},
        }
        
        return [lines]

                
def get_tpc_boundary_plot(cryoid=0,tpcid=0,color=(0,0,0),use_tick_dimensions=False):

    try:
        import ROOT as rt
        from larlite import larlite
        from larlite import larutil
    except:
        raise ValueError("Could not load ROOT, larlite, or larlite::larutil")

    detid = larutil.LArUtilConfig.Detector()

    if detid == larlite.geo.kDetIdMax:
        print("Detector Configuration not set")
        print("Call larutil.LArUtilConfig.SetDetector( DetId_t )")
        return None

    minbound = rt.TVector3()
    maxbound = rt.TVector3()

    larlite.larutil.Geometry.GetME().TPCBoundaries(minbound, maxbound, tpcid, cryoid )
    if use_tick_dimensions:
        from ROOT import larutil
        nticks = float(larutil.DetectorProperties.GetME().ReadOutWindowSize())
        tpcdriftdir = larlite.larutil.Geometry.GetME().TPCDriftDir(tpcid,cryoid)
        if tpcdriftdir[0]>0:
            minbound[0] = 0
            maxbound[0] = nticks
        else:
            minbound[0] = -nticks
            maxbound[0] = 0

    #print(bounds)
    top_pts  = [ [minbound[0],maxbound[1], minbound[2]],
                 [maxbound[0],maxbound[1], minbound[2]],
                 [maxbound[0],maxbound[1], maxbound[2]],
                 [minbound[0],maxbound[1], maxbound[2]],
                 [minbound[0],maxbound[1], minbound[2]] ]
    bot_pts  = [ [minbound[0],minbound[1], minbound[2]],
                 [maxbound[0],minbound[1], minbound[2]],
                 [maxbound[0],minbound[1], maxbound[2]],
                 [minbound[0],minbound[1], maxbound[2]],
                 [minbound[0],minbound[1], minbound[2]] ]
    up_pts   = [ [minbound[0],minbound[1], minbound[2]],
                 [maxbound[0],minbound[1], minbound[2]],
                 [maxbound[0],maxbound[1], minbound[2]],
                 [minbound[0],maxbound[1], minbound[2]],
                 [minbound[0],minbound[1], minbound[2]] ]
    down_pts = [ [minbound[0],minbound[1], maxbound[2]],
                 [maxbound[0],minbound[1], maxbound[2]],
                 [maxbound[0],maxbound[1], maxbound[2]],
                 [minbound[0],maxbound[1], maxbound[2]],
                 [minbound[0],minbound[1], maxbound[2]] ]        

            
    Xe = []
    Ye = []
    Ze = []

    for boundary in [top_pts,bot_pts,up_pts,down_pts]:
        for ipt, pt in enumerate(boundary):
            Xe.append( pt[0] )
            Ye.append( pt[1] )
            Ze.append( pt[2] )
        Xe.append(None)
        Ye.append(None)
        Ze.append(None)
    lines = {
        "type": "scatter3d",
        "x": Xe,
        "y": Ye,
        "z": Ze,
        "mode": "lines",
        "name": "TPC(%d,%d)"%(cryoid,tpcid),
        "line": {"color": "rgb(%d,%d,%d)"%color, "width": 5},
    }

    return lines    

