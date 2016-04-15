

#include "eigen3/Eigen/Core"
#include <omp.h>
#include <iostream>

double sample3D(unsigned char * A, int I, int J, int K, const Eigen::Vector3d& p)
{
  // Sample a 3D array - linear interpolation?
  const int i = std::max(0, std::min(static_cast<int>(p[0]), I-2));
  const int j = std::max(0, std::min(static_cast<int>(p[1]), J-2));
  const int k = std::max(0, std::min(static_cast<int>(p[2]), K-2));
  const double x = std::min(p[0] - i, 1.0);
  const double y = std::min(p[1] - j, 1.0);
  const double z = std::min(p[2] - k, 1.0);
  const int idx = K*(J*i + j) + k;
  const double V000 = A[idx];
  const double V001 = A[idx+1];
  const double V010 = A[idx+K];
  const double V011 = A[idx+K+1];
  const double V100 = A[idx];
  const double V101 = A[idx+J*K+1];
  const double V110 = A[idx+J*K+K];
  const double V111 = A[idx+J*K+K+1];
  return V000*(1-x)*(1-y)*(1-z) + V100*x*(1-y)*(1-z) +
    V010*(1-x)*y*(1-z) + V001*(1-x)*(1-y)*z +
    V101*x*(1-y)*z + V011*(1-x)*y*z +
    V110*x*y*(1-z) + V111*x*y*z;  
}

extern "C" void _mesh_reproject(unsigned char* A, int I, int J, int K, double* spacing, double* verts, double* norms,  int NV, unsigned int* tris, int NT, double level, double t, double s)
{
  const Eigen::Vector3d bbox0(0, 0, 0);
  const Eigen::Vector3d bbox1(I-1, J-1, K-1);
  Eigen::Vector3d sp = Eigen::Map<Eigen::Vector3d>(spacing);
  //  Eigen::VectorXd u(NV);
  //  u.setZero();
  
  #pragma omp for
  for(int i=0; i<NV; i++)
    {
      Eigen::Map<Eigen::Vector3d> v(&verts[3*i]);
      Eigen::Map<Eigen::Vector3d> n(&norms[3*i]);
      Eigen::Vector3d p2 = ((v.cwiseQuotient(sp)).cwiseMax(bbox0)).cwiseMin(bbox1);
      double l = sample3D(A, I, J, K, p2);

      //      std::cout << v.transpose() << " " << l << " " << level << " " << t << " " << n.transpose();
      v += t*exp(-(l/level))*n;
      //std::cout << " " << v.transpose() <<"\n";
    }
}


extern "C" void _mesh_project_mean(unsigned char* A, int I, int J, int K, double* spacing, double* verts, double* norms, double* signal, int NV, double d0, double d1, int ns)
{
  const Eigen::Vector3d bbox0(0, 0, 0);
  const Eigen::Vector3d bbox1(I-1, J-1, K-1);
  Eigen::Vector3d sp = Eigen::Map<Eigen::Vector3d>(spacing);
  #pragma omp for
  for(int i=0; i<NV; i++)
    {
      Eigen::Map<Eigen::Vector3d> v(&verts[3*i]);
      Eigen::Map<Eigen::Vector3d> n(&norms[3*i]);
      Eigen::Vector3d p_start = (v + d0*n);
      Eigen::Vector3d p_start2 = ((p_start.cwiseQuotient(sp)).cwiseMax(bbox0)).cwiseMin(bbox1);
      Eigen::Vector3d p_end = (v + d1*n);
      Eigen::Vector3d p_end2 = ((p_end.cwiseQuotient(sp)).cwiseMax(bbox0)).cwiseMin(bbox1);
      double tot_signal = 0.0;

      for(int j=0; j<ns; j++)
	{
	  double x = static_cast<double>(j)/ns;
	  tot_signal += sample3D(A, I, J, K, p_start2*(1.0-x) + p_end2*x);
	}
      signal[i] = tot_signal / ns;
    }
}
