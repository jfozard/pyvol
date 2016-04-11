
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <map>
#include <array>
#include <cstring>

/*
#include "boost/tuple/tuple.hpp"
#include "boost/tuple/tuple_comparison.hpp"
*/

typedef unsigned char uchar;
typedef std::array<float, 3> point;
typedef std::array<int, 3> Tri;

class pointMap {
 public:
  std::map<point, int> m;
  std::vector<point> v;
  typedef std::map<point, int> m_type;
  typedef m_type::iterator mi_type;
  int getIdx(point p)
  {
    std::pair<mi_type, bool> const& r=m.insert(m_type::value_type(p, m.size()));
    if (r.second) { 
      v.push_back(p);
    } else {
    // value wasn't inserted because my_map[foo_obj] already existed.
    // note: the old value is available through r.first->second
    }
    return r.first->second;
  } 
};


extern "C" void img_make_iso(uchar* img, int NX, int NY, int NZ, uchar level, float** verts, int** tris, int* NV, int* NT)
{

  std::vector<Tri> tl;
  const int sz = 1;
  const int sy = NZ;
  const int sx = NY*NZ;

  pointMap pm;
  for(int lx=0; lx<NX-1; ++lx)
    for(int ly=0; ly<NY; ++ly)
      for(int lz=0; lz<NZ; ++lz)
	{
	bool l0 = img[lz*sz+ly*sy+lx*sx]<level;
        bool l1 = img[lz*sz+ly*sy+(lx+1)*sx]<level;
	if (l0!=l1)
	  {
	    int i0 = pm.getIdx({lx+1, ly, lz});
	    int i1 = pm.getIdx({lx+1, ly+1, lz});
	    int i2 = pm.getIdx({lx+1, ly, lz+1});
	    int i3 = pm.getIdx({lx+1, ly+1, lz+1}); 
	    if(!l0)
	      {
		tl.push_back({i0, i1, i2});
		tl.push_back({i1, i3, i2});
	      }
	    if(!l1)
	      {
		tl.push_back({i1, i0, i2});
		tl.push_back({i1, i2, i3});
	      }
	  }
	}
  for(int lx=0; lx<NX; ++lx)
    for(int ly=0; ly<NY-1; ++ly)
      for(int lz=0; lz<NZ; ++lz)
	{
	  bool l0 = img[lz*sz+ly*sy+lx*sx]<level;
	  bool l1 = img[lz*sz+(ly+1)*sy+lx*sx]<level;
	  if(l0!=l1)
	    {
	      int i0 = pm.getIdx({lx, ly+1, lz});
	      int i1 = pm.getIdx({lx+1, ly+1, lz});
	      int i2 = pm.getIdx({lx, ly+1, lz+1});
	      int i3 = pm.getIdx({lx+1, ly+1, lz+1}); 
	      if(!l0)
		{
		  tl.push_back({i0, i2, i1});
		  tl.push_back({i1, i2, i3});
		}
	      if(!l1)
		{
		  tl.push_back({i2, i0, i1});
		  tl.push_back({i2, i1, i3});
		}
	    }
	}
  for(int lx=0; lx<NX; ++lx)
    for(int ly=0; ly<NY; ++ly)
      for(int lz=0; lz<NZ-1; ++lz)
	{
	  bool l0 = img[lz*sz+ly*sy+lx*sx]<level;
	  bool l1 = img[(lz+1)*sz+ly*sy+lx*sx]<level;
	  if(l0!=l1)
	    {
	      int i0 = pm.getIdx({lx, ly, lz+1});
	      int i1 = pm.getIdx({lx+1, ly, lz+1});
	      int i2 = pm.getIdx({lx, ly+1, lz+1});
	      int i3 = pm.getIdx({lx+1, ly+1, lz+1}); 
	      if(!l0) {
		tl.push_back({i0, i1, i2});
		tl.push_back({i1, i3, i2});
	      }
	      if(!l1) {
		tl.push_back({i1, i0, i2});
		tl.push_back({i1, i2, i3});
	      }
	    }
	}


  *NV = pm.v.size();
  *NT = tl.size();
  float* new_verts = (float*) std::malloc(sizeof(point)*pm.v.size());
  int* new_tris = (int*) std::malloc(sizeof(Tri)*tl.size());

  std::memcpy(new_verts, pm.v.data(), sizeof(point)*pm.v.size());
  std::memcpy(new_tris, tl.data(), sizeof(Tri)*tl.size());

  *verts = new_verts;
  *tris = new_tris;

  /*
  f << "ply\nformat ascii 1.0\n";
  int Np = pm.v.size();
  f << "element vertex " << Np <<"\n";
  f << "property float x\n";
  f << "property float y\n";
  f << "property float z\n";
  int Nt = tl.size();
  f << "element face " << Nt << "\n";
  f << "property list uchar int vertex_indices\n";
  f << "property uint label\n";
  f << "end_header\n";  
  for(int i=0; i<Np; ++i)
    {
      f << pm.v[i].get<0>() << " " << pm.v[i].get<1>() << " " << pm.v[i].get<2>() << "\n";
    }
  for(int i=0; i<Nt; ++i)
    {
      f << "3 " << tl[i].get<1>() << " " << tl[i].get<2>() << " " << tl[i].get<3>() << " " << tl[i].get<0>() << "\n";
    }
  */
}

