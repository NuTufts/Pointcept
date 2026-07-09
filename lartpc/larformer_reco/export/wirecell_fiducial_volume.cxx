
#include <math.h>
#include <vector>


class WCFiducial {

  private:

    double m_top = 117;
    double m_bottom = -116;
    double m_upstream = 0.;
    double m_downstream = 1037.;
    double m_anode = 0.;
    double m_cathode = 256.;

    double m_sc_bottom_1_y=-116.;
    double m_sc_bottom_1_x=80.;
    double m_sc_bottom_2_y=-99.;
    double m_sc_bottom_2_x=256.;
    double m_sc_top_1_y = 116.; // used to be 118 cm
    double m_sc_top_1_x = 100.;
    double m_sc_top_2_y = 102.; // used to be 103 cm
    double m_sc_top_2_x = 256.;
    double m_sc_upstream_1_z = 0.;
    double m_sc_upstream_1_x = 120.;
    double m_sc_upstream_2_z = 11.;
    double m_sc_upstream_2_x = 256.;
    double m_sc_downstream_1_z=1037.;
    double m_sc_downstream_1_x=120.;
    double m_sc_downstream_2_z=1026.;
    double m_sc_downstream_2_x=256.;

    std::vector<double> boundary_xy_x, boundary_xy_y;
    std::vector<double> boundary_xz_x, boundary_xz_z;

    std::vector<std::vector<double>> boundary_xy_x_array, boundary_xy_y_array;
    std::vector<std::vector<double>> boundary_xz_x_array, boundary_xz_z_array;

    int pnpoly(std::vector<double>& vertx, std::vector<double>& verty, double testx, double testy){
      int i, j, c = 0;
      for (i = 0, j = int(vertx.size())-1; i < int(vertx.size()); j = i++) {
        if ( ((verty[i]>testy) != (verty[j]>testy)) &&
             (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
          c = !c;
      }
      return c;
    }


    void initialize_arrays(double boundary_dis_cut){

      boundary_xy_x.clear(); boundary_xy_y.clear();
      boundary_xy_x.push_back(m_anode + boundary_dis_cut);
      boundary_xy_y.push_back(m_bottom + boundary_dis_cut);
      boundary_xy_x.push_back(m_sc_bottom_1_x-boundary_dis_cut);
      boundary_xy_y.push_back(m_sc_bottom_1_y+boundary_dis_cut);
      boundary_xy_x.push_back(m_sc_bottom_2_x-boundary_dis_cut);
      boundary_xy_y.push_back(m_sc_bottom_2_y+boundary_dis_cut);
      boundary_xy_x.push_back(m_sc_top_2_x-boundary_dis_cut);
      boundary_xy_y.push_back(m_sc_top_2_y-boundary_dis_cut);
      boundary_xy_x.push_back(m_sc_top_1_x-boundary_dis_cut);
      boundary_xy_y.push_back(m_sc_top_1_y-boundary_dis_cut);
      boundary_xy_x.push_back(m_anode + boundary_dis_cut);
      boundary_xy_y.push_back(m_top - boundary_dis_cut);

      boundary_xz_x.clear(); boundary_xz_z.clear();
      boundary_xz_x.push_back(m_anode + boundary_dis_cut);
      boundary_xz_z.push_back(m_upstream + boundary_dis_cut+1.);
      boundary_xz_x.push_back(m_sc_upstream_1_x - boundary_dis_cut);
      boundary_xz_z.push_back(m_sc_upstream_1_z + boundary_dis_cut+1.);
      boundary_xz_x.push_back(m_sc_upstream_2_x - boundary_dis_cut);
      boundary_xz_z.push_back(m_sc_upstream_2_z + boundary_dis_cut+1.);
      boundary_xz_x.push_back(m_sc_downstream_2_x - boundary_dis_cut);
      boundary_xz_z.push_back(m_sc_downstream_2_z - boundary_dis_cut-1.);
      boundary_xz_x.push_back(m_sc_downstream_1_x - boundary_dis_cut);
      boundary_xz_z.push_back(m_sc_downstream_1_z - boundary_dis_cut-1.);
      boundary_xz_x.push_back(m_anode + boundary_dis_cut);
      boundary_xz_z.push_back(m_downstream - boundary_dis_cut-1.);

    }

  public:

    WCFiducial(){ initialize_arrays(3.); }
    WCFiducial(double boundary){ initialize_arrays(boundary); }

    bool insideFV(double x, double y, double z){

      int c1=0;
      int c2=0;
      int index_y = floor((y+116)/24);
      int index_z = floor(z/100.);
      if(index_y<0){index_y=0;} else if(index_y>9){index_y=9;}
      if(index_z<0){index_z=0;} else if(index_z>9){index_z=9;}
      c1 = pnpoly(boundary_xy_x, boundary_xy_y, x, y);
      c2 = pnpoly(boundary_xz_x, boundary_xz_z, x, z);
  
      if (c1 && c2) return true;
      return false;

    }

};


extern "C" {

  WCFiducial* WCFiducialCustom_new(double b){
    return new WCFiducial(b);
  }
  WCFiducial* WCFiducial_new(){
    return new WCFiducial();
  }
  bool WCFiducial_insideFV(WCFiducial* wcfid, double x, double y, double z){
    return wcfid->insideFV(x, y, z);
  }

}


