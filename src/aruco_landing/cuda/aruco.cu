// Experimental CUDA ArUco 4x4 detector; no host image/contour processing.
// C ABI intentionally avoids a dependency on a particular PyTorch C++ ABI.
#include <cuda_runtime.h>
#include <cmath>
#include <climits>

struct Workspace { int n,h,w,cap,pixels; int *parent,*stats,*candidates,*candidate_count; unsigned char* gray; };
__device__ int root(int* p,int i) {
  while(p[i]!=i) { int next=p[i]; atomicMin(p+i,p[next]); i=next; }
  return i;
}
__global__ void init(Workspace a,const unsigned char* rgb,int channels) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=a.pixels)return;
  int v=(77*rgb[i*channels]+150*rgb[i*channels+1]+29*rgb[i*channels+2])>>8;
  a.gray[i]=v; a.parent[i]=v<127?i:-1;
  a.stats[i*5]=INT_MAX; a.stats[i*5+1]=INT_MAX;
  a.stats[i*5+2]=-1; a.stats[i*5+3]=-1; a.stats[i*5+4]=0;
}
__device__ void join(int* p,int a,int b) {
  for(;;) { a=root(p,a);b=root(p,b); if(a==b)return; int hi=max(a,b),lo=min(a,b);
    if(atomicCAS(p+hi,hi,lo)==hi)return; }
}
__global__ void unite(Workspace a) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=a.pixels||a.parent[i]<0)return;
  int x=i%a.w,y=(i/a.w)%a.h;
  if(x>0&&a.parent[i-1]>=0)join(a.parent,i,i-1);
  if(y>0&&a.parent[i-a.w]>=0)join(a.parent,i,i-a.w);
}
__global__ void stats(Workspace a) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=a.pixels||a.parent[i]<0)return;
  int r=root(a.parent,i); a.parent[i]=r; int *s=a.stats+r*5;
  int x=i%a.w,y=(i/a.w)%a.h;
  atomicMin(s,x);atomicMin(s+1,y);atomicMax(s+2,x);atomicMax(s+3,y);atomicAdd(s+4,1);
}
__global__ void candidates(Workspace a) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=a.pixels)return;
  int *s=a.stats+i*5; int dx=s[2]-s[0],dy=s[3]-s[1];
  if(s[4]<40||dx<12||dy<12||dx>400||dy>400||dx>4*dy||dy>4*dx||
     s[0]<2||s[1]<2||s[2]>=a.w-2||s[3]>=a.h-2)return;
  int env=i/(a.w*a.h), slot=atomicAdd(a.candidate_count+env,1);
  if(slot<a.cap)a.candidates[env*a.cap+slot]=i;
}
__device__ float sample(const unsigned char* im,int w,int h,float x,float y) {
  if(x<0||y<0||x>=w-1||y>=h-1)return 255;
  int ix=(int)x,iy=(int)y; float u=x-ix,v=y-iy;
  return (1-u)*(1-v)*im[iy*w+ix]+u*(1-v)*im[iy*w+ix+1]+(1-u)*v*im[(iy+1)*w+ix]+u*v*im[(iy+1)*w+ix+1];
}
__global__ void decode(Workspace a,const int* codes,int code_count,float* out,int* count) {
  int slot=blockIdx.x,env=slot/a.cap; if(threadIdx.x||slot%a.cap>=a.candidate_count[env]||a.candidate_count[env]>a.cap)return;
  int r=a.candidates[slot],*s=a.stats+r*5,base=env*a.w*a.h;
  float qx[4],qy[4];int bestmin=INT_MAX,bestmax=-1;
  for(int y=s[1];y<=s[3];y++)for(int x=s[0];x<=s[2];x++) {
    if(a.parent[base+y*a.w+x]!=r)continue;
    int sum=x+y;
    if(sum<bestmin){bestmin=sum;qx[0]=x;qy[0]=y;}
    if(sum>=bestmax){bestmax=sum;qx[2]=x;qy[2]=y;}
  }
  float dx=qx[2]-qx[0],dy=qy[2]-qy[0],lo=0,hi=0;
  for(int y=s[1];y<=s[3];y++)for(int x=s[0];x<=s[2];x++) {
    if(a.parent[base+y*a.w+x]!=r)continue;
    float v=dx*(y-qy[0])-dy*(x-qx[0]);
    if(v<lo){lo=v;qx[1]=x;qy[1]=y;} if(v>hi){hi=v;qx[3]=x;qy[3]=y;}
  }
  if(lo>=-16||hi<=16)return;
  // Fit the outer border lines to component boundary pixels. TLS line fits
  // average the raster staircase and give subpixel line intersections.
  float nx[4],ny[4],rho[4];
  for(int edge=0;edge<4;edge++) {
    int next=(edge+1)%4;float ex=qx[next]-qx[edge],ey=qy[next]-qy[edge];
    float len=hypotf(ex,ey);if(len<12)return;
    float ax=ey/len,ay=-ex/len,ar=ax*qx[edge]+ay*qy[edge];
    double sx=0,sy=0,sxx=0,sxy=0,syy=0;int num=0;
    for(int y=s[1];y<=s[3];y++)for(int x=s[0];x<=s[2];x++) {
      int i=base+y*a.w+x;if(a.parent[i]!=r)continue;
      if(a.parent[i-1]==r&&a.parent[i+1]==r&&a.parent[i-a.w]==r&&a.parent[i+a.w]==r)continue;
      float along=((x-qx[edge])*ex+(y-qy[edge])*ey)/len;
      if(along<2||along>len-2||fabsf(ax*x+ay*y-ar)>1.2f)continue;
      sx+=x;sy+=y;sxx+=x*x;sxy+=x*y;syy+=y*y;num++;
    }
    if(num<6)return;
    double mx=sx/num,my=sy/num;
    float angle=.5f*atan2f(2*(sxy/num-mx*my),(sxx/num-mx*mx)-(syy/num-my*my));
    nx[edge]=-sinf(angle);ny[edge]=cosf(angle);
    if(nx[edge]*ax+ny[edge]*ay<0){nx[edge]=-nx[edge];ny[edge]=-ny[edge];}
    rho[edge]=nx[edge]*mx+ny[edge]*my+.5f;
  }
  for(int j=0;j<4;j++) {
    int k=(j+3)%4;float den=nx[k]*ny[j]-nx[j]*ny[k];if(fabsf(den)<.1)return;
    qx[j]=(rho[k]*ny[j]-rho[j]*ny[k])/den;
    qy[j]=(nx[k]*rho[j]-nx[j]*rho[k])/den;
  }
  float dx1=qx[1]-qx[2],dx2=qx[3]-qx[2],dx3=qx[0]-qx[1]+qx[2]-qx[3];
  float dy1=qy[1]-qy[2],dy2=qy[3]-qy[2],dy3=qy[0]-qy[1]+qy[2]-qy[3];
  float den=dx1*dy2-dx2*dy1;if(fabsf(den)<1)return;
  float g=(dx3*dy2-dx2*dy3)/den,h=(dx1*dy3-dx3*dy1)/den;
  float A=qx[1]-qx[0]+g*qx[1],B=qx[3]-qx[0]+h*qx[3];
  float C=qy[1]-qy[0]+g*qy[1],D=qy[3]-qy[0]+h*qy[3];
  int bits=0;
  for(int row=0;row<6;row++)for(int col=0;col<6;col++) {
    float mean=0;
    for(int sy=-1;sy<=1;sy++)for(int sx=-1;sx<=1;sx++) {
      float u=(col+.5f+sx*.2f)/6,v=(row+.5f+sy*.2f)/6,z=g*u+h*v+1;
      mean+=sample(a.gray+base,a.w,a.h,(A*u+B*v+qx[0])/z,(C*u+D*v+qy[0])/z);
    }
    bool white=mean>127*9;
    if(row==0||row==5||col==0||col==5){if(white)return;}
    else if(white)bits|=1<<((row-1)*4+col-1);
  }
  int winner=-1,distance=2,ties=0;
  for(int k=0;k<code_count;k++) {int d=__popc(bits^codes[k]);
    if(d<distance){distance=d;winner=k;ties=1;}else if(d==distance)ties++;}
  if(winner<0||ties!=1)return;
  int index=atomicAdd(count+env,1);if(index>=a.cap)return;
  float *result=out+(env*a.cap+index)*9;result[0]=winner/4;
  for(int j=0;j<4;j++){int k=(j+4-winner%4)%4;result[1+2*j]=qx[k];result[2+2*j]=qy[k];}
}
extern "C" void aruco_destroy(Workspace* a) {
  if(!a)return;cudaFree(a->parent);cudaFree(a->stats);cudaFree(a->gray);cudaFree(a->candidates);cudaFree(a->candidate_count);delete a;
}
extern "C" Workspace* aruco_create(int n,int h,int w,int cap) {
  Workspace *a=new Workspace{n,h,w,cap,n*h*w,nullptr,nullptr,nullptr,nullptr,nullptr};
  if(cudaMalloc(&a->parent,a->pixels*sizeof(int))!=cudaSuccess||cudaMalloc(&a->stats,a->pixels*5*sizeof(int))!=cudaSuccess||
     cudaMalloc(&a->gray,a->pixels)!=cudaSuccess||cudaMalloc(&a->candidates,n*cap*sizeof(int))!=cudaSuccess||
     cudaMalloc(&a->candidate_count,n*sizeof(int))!=cudaSuccess){aruco_destroy(a);return nullptr;}return a;
}
__global__ void overflow(Workspace a,int* count) {
  int env=blockIdx.x*blockDim.x+threadIdx.x;
  if(env<a.n&&a.candidate_count[env]>a.cap)count[env]=-1;
}
extern "C" int aruco_detect(Workspace* a,const unsigned char* rgb,float* output,int* count,const int* codes,int channels,cudaStream_t stream) {
  cudaMemsetAsync(count,0,a->n*sizeof(int),stream);cudaMemsetAsync(a->candidate_count,0,a->n*sizeof(int),stream);
  int blocks=(a->pixels+255)/256;
  init<<<blocks,256,0,stream>>>(*a,rgb,channels);unite<<<blocks,256,0,stream>>>(*a);
  stats<<<blocks,256,0,stream>>>(*a);candidates<<<blocks,256,0,stream>>>(*a);
  decode<<<a->n*a->cap,1,0,stream>>>(*a,codes,400,output,count);
  overflow<<<(a->n+63)/64,64,0,stream>>>(*a,count);
  // Overflow must be visible to the caller, never silently truncate markers.
  return cudaGetLastError();
}
