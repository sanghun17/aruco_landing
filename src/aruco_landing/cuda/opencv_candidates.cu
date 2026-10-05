// SPDX-License-Identifier: Apache-2.0
// CUDA adaptation of OpenCV 4.13.0 contours_new.cpp, approx.cpp and the
// candidate grouping in aruco_detector.cpp. See opencv-LICENSE and
// opencv-port.md for upstream provenance. Other paths in aruco.cu are unchanged.
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

__device__ const int dx8[8]={1,1,0,-1,-1,-1,0,1};
__device__ const int dy8[8]={0,-1,-1,-1,0,1,1,1};

// SAT is built over BORDER_REPLICATE padding on the caller's CUDA stream.
__global__ void threshold_kernel(const int* sat, signed char* binary,int n,int h,int w,
                                int pad,int lo,int step,int levels,double constant) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, plane=i/((h+2)*(w+2));
    if(plane>=n*levels)return;
    int y=(i/ (w+2))%(h+2),x=i%(w+2),env=plane/levels,level=plane%levels;
    if(!x||!y||x==w+1||y==h+1){binary[i]=0;return;}
    int k=lo+level*step;if(!(k&1))k++;
    int radius=k/2,sw=w+2*pad+1,sh=h+2*pad+1;
    int x0=x-1+pad-radius,y0=y-1+pad-radius;
    const int* a=sat+env*sw*sh;
    int sum=a[(y0+k)*sw+x0+k]-a[y0*sw+x0+k]-a[(y0+k)*sw+x0]+a[y0*sw+x0];
    int mean=__double2int_rn(double(sum)/double(k*k));
    // Original gray pixel is the 1x1 SAT difference at its padded position.
    int xx=x-1+pad,yy=y-1+pad;
    int gray=a[(yy+1)*sw+xx+1]-a[yy*sw+xx+1]-a[(yy+1)*sw+xx]+a[yy*sw+xx];
    binary[i]=(gray-mean<=-int(floor(constant)))?1:0;
}

// Closed Douglas-Peucker, including OpenCV 4.13's segment-distance test
// and final straight-line cleanup. Integer contour points preserve tie order.
__device__ int approximate(int2* src,int count,double epsilon,int2* dst,int2* stack) {
    if(!count)return 0;
    double eps=epsilon*epsilon;int pos=0,far=0,start=0,end=0,top=0,out=0;
    int2 a,b,p;bool small=false;
    for(int iteration=0;iteration<3;iteration++) {
        pos=(pos+far)%count;a=src[pos];pos=(pos+1)%count;double greatest=0;
        for(int j=1;j<count;j++) {
            p=src[pos];pos=(pos+1)%count;
            double x=p.x-a.x,y=p.y-a.y,d=x*x+y*y;
            if(d>greatest){greatest=d;far=j;}
        }
        small=greatest<=eps;
    }
    if(small)dst[out++]=a;
    else {
        start=pos%count;end=(far+start)%count;
        stack[top++]=make_int2(end,start);stack[top++]=make_int2(start,end);
    }
    while(top) {
        int2 slice=stack[--top];b=src[slice.y];pos=slice.x;a=src[pos];pos=(pos+1)%count;
        double x=b.x-a.x,y=b.y-a.y,len=x*x+y*y,maxd=0;
        if(pos!=slice.y) {
            while(pos!=slice.y) {
                p=src[pos];pos=(pos+1)%count;
                double ax=p.x-a.x,ay=p.y-a.y,projection=ax*x+ay*y,d;
                if(projection<0)d=(ax*ax+ay*ay)*len;
                else if(projection>len){double bx=p.x-b.x,by=p.y-b.y;d=(bx*bx+by*by)*len;}
                else {double cross=ay*x-ax*y;d=cross*cross;}
                if(d>maxd){maxd=d;far=(pos+count-1)%count;}
            }
            small=maxd<=eps*len;
        }else small=true;
        if(small)dst[out++]=a;
        else {stack[top++]=make_int2(far,slice.y);stack[top++]=make_int2(slice.x,far);}
    }
    count=out;pos=count-1;a=dst[pos];pos=(pos+1)%count;
    int wpos=pos;p=dst[pos];pos=(pos+1)%count;
    for(int i=0;i<count&&out>2;i++) {
        b=dst[pos];pos=(pos+1)%count;double x=b.x-a.x,y=b.y-a.y;
        double dist=fabs((p.x-a.x)*y-(p.y-a.y)*x);
        double product=double(p.x-a.x)*(b.x-p.x)+double(p.y-a.y)*(b.y-p.y);
        if(dist*dist<=.5*eps*(x*x+y*y)&&x!=0&&y!=0&&product>=0) {
            out--;dst[wpos]=a=b;wpos=(wpos+1)%count;p=dst[pos];pos=(pos+1)%count;i++;continue;
        }
        dst[wpos]=a=p;wpos=(wpos+1)%count;p=b;
    }
    dst[wpos]=p;return out;
}

// One scanner per image/threshold scale. Sequential border following preserves
// RETR_LIST / CHAIN_APPROX_NONE semantics; independent planes run concurrently.
__global__ void contour_kernel(signed char* binary,int h,int w,int planes,
                              int2* work,int work_size,float* quads,int* counts,int cap,
                              double min_rate,double max_rate,double accuracy,double min_corner) {
    int plane=blockIdx.x;if(threadIdx.x||plane>=planes)return;
    int stride=w+2;signed char* im=binary+plane*(h+2)*stride;
    int2* points=work+plane*work_size*3;int2* poly=points+work_size;int2* stack=poly+work_size;
    int minp=int(min_rate*max(w,h)),maxp=int(max_rate*max(w,h)),out=0;
    for(int y=1;y<=h;y++) {
        int prev=0;
        for(int x=1;x<=w;x++) {
            int value=im[y*stride+x];if(value==prev)continue;
            bool external=prev==0&&value==1;
            bool hole=!external&&value==0&&prev>=1;
            if(!external&&!hole){prev=value;continue;}
            int px=x-(hole?1:0),py=y,origin=py*stride+px;
            int direction=hole?0:4,end_direction=direction,first=origin;
            do {direction=(direction-1)&7;first=origin+dx8[direction]+dy8[direction]*stride;}
            while(!im[first]&&direction!=end_direction);
            int size=0;
            if(direction==end_direction) {
                im[origin]=static_cast<signed char>(0x82);points[size++]=make_int2(px-1,py-1);
            }else {
                int current=origin,next;
                for(;;) {
                    int old=direction;
                    do {++direction;next=current+dx8[direction&7]+dy8[direction&7]*stride;}
                    while(!im[next]&&direction<15);
                    direction&=7;
                    if(unsigned(direction-1)<unsigned(old))im[current]=static_cast<signed char>(0x82);
                    else if(im[current]==1)im[current]=2;
                    if(size<work_size)points[size]=make_int2(px-1,py-1);size++;
                    px+=dx8[direction];py+=dy8[direction];
                    if(next==origin&&current==first)break;
                    current=next;direction=(direction+4)&7;
                    if(size>4*(h+2)*(w+2)){counts[plane]=-2;return;}
                }
            }
            if(size>=minp&&size<=maxp&&size<=work_size) {
                int count=approximate(points,size,accuracy*size,poly,stack);
                if(count==4) {
                    int sign=0;bool convex=true;double mind=double(max(h,w))*max(h,w);
                    for(int j=0;j<4;j++) {
                        int2 a=poly[j],b=poly[(j+1)%4],c=poly[(j+2)%4];
                        long long cross=(long long)(b.x-a.x)*(c.y-b.y)-(long long)(b.y-a.y)*(c.x-b.x);
                        int s=(cross>0)-(cross<0);if(!s||(sign&&s!=sign))convex=false;sign=s;
                        double dx=b.x-a.x,dy=b.y-a.y;mind=fmin(mind,dx*dx+dy*dy);
                    }
                    if(convex&&mind>=size*size*min_corner*min_corner) {
                        if(out>=cap){counts[plane]=-1;return;}
                        // OpenCV reorders to clockwise after approximation.
                        if(sign<0){int2 tmp=poly[1];poly[1]=poly[3];poly[3]=tmp;}
                        float* q=quads+(plane*cap+out)*8;
                        for(int j=0;j<4;j++){q[2*j]=poly[j].x;q[2*j+1]=poly[j].y;}
                        out++;
                    }
                }
            }
            // A resumed scanner reads the *marked* preceding pixel.
            prev=im[y*stride+x];
        }
    }
    counts[plane]=out;
}

extern "C" int ocv_threshold(const int* sat,signed char* binary,int n,int h,int w,
 int pad,int lo,int step,int levels,double constant,cudaStream_t stream) {
    threshold_kernel<<<(n*levels*(h+2)*(w+2)+255)/256,256,0,stream>>>(sat,binary,n,h,w,pad,lo,step,levels,constant);
    return cudaGetLastError();
}
extern "C" int ocv_candidates(signed char* binary,int h,int w,int planes,int2* work,int work_size,
 float* quads,int* counts,int cap,double min_rate,double max_rate,double accuracy,double min_corner,cudaStream_t stream) {
    contour_kernel<<<planes,1,0,stream>>>(binary,h,w,planes,work,work_size,quads,counts,cap,min_rate,max_rate,accuracy,min_corner);
    return cudaGetLastError();
}

struct Candidate {int original,parent=-1,depth=0;float q[8],perimeter=0;std::vector<int> close;};
static float distance(const Candidate& a,const Candidate& b) {
    float best=INFINITY;
    for(int r=0;r<4;r++){float sum=0;for(int k=0;k<4;k++) {
        float x=a.q[2*((k+r)%4)]-b.q[2*k],y=a.q[2*((k+r)%4)+1]-b.q[2*k+1];sum+=x*x+y*y;
    }best=std::min(best,sum*.25f);}return std::sqrt(best);
}
static bool inside(const Candidate& a,const Candidate& b) {
    for(int k=0;k<4;k++){int sign=0;for(int j=0;j<4;j++) {
        double x=b.q[2*((j+1)%4)]-b.q[2*j],y=b.q[2*((j+1)%4)+1]-b.q[2*j+1];
        double cross=x*(a.q[2*k+1]-b.q[2*j+1])-y*(a.q[2*k]-b.q[2*j]);
        int s=(cross>0)-(cross<0);if(s&&sign&&s!=sign)return false;if(s)sign=s;
    }}return true;
}
// Compact CPU control stage follows OpenCV's stable grouping and hierarchy.
// It receives quadrilaterals, never full images or threshold rasters.
extern "C" int ocv_group(const float* quads,int count,int h,int w,int marker_size,int border,
 double min_distance,int border_distance,double group_distance,int inverted,int* metadata,int* close_indices) {
    std::vector<Candidate> c(count);
    for(int i=0;i<count;i++){c[i].original=i;std::copy(quads+i*8,quads+(i+1)*8,c[i].q);
        for(int j=0;j<4;j++){float x=c[i].q[2*j]-c[i].q[2*((j+1)%4)],y=c[i].q[2*j+1]-c[i].q[2*((j+1)%4)+1];c[i].perimeter+=std::sqrt(x*x+y*y);}}
    std::stable_sort(c.begin(),c.end(),[](const Candidate&a,const Candidate&b){return a.perimeter>b.perimeter;});
    std::vector<int> group(count,-1);std::vector<bool> selected(count,true);std::vector<std::vector<int>> groups;
    for(int i=0;i<count;i++) {
        for(int j=i+1;j<count;j++)if(distance(c[i],c[j])<c[j].perimeter*float(min_distance)) {
            selected[i]=selected[j]=false;
            if(group[i]<0&&group[j]<0){group[i]=group[j]=groups.size();groups.push_back({i,j});}
            else if(group[i]>=0&&group[j]<0){group[j]=group[i];groups[group[i]].push_back(j);}
            else if(group[j]>=0&&group[i]<0){group[i]=group[j];groups[group[j]].push_back(i);}
        }
        if(selected[i]){selected[i]=false;group[i]=groups.size();groups.push_back({i});}
    }
    for(auto& g:groups) {
        std::stable_sort(g.begin(),g.end());if(inverted)std::reverse(g.begin(),g.end());int current=g[0];bool near=false;
        for(int j=0;j<4;j++)if(c[current].q[2*j]<border_distance||c[current].q[2*j+1]<border_distance||
           c[current].q[2*j]>w-1-border_distance||c[current].q[2*j+1]>h-1-border_distance)near=true;
        if(near)continue;selected[current]=true;
        for(unsigned j=1;j<g.size();j++) {
            int id=g[j];float module=c[id].perimeter/(4*(marker_size+2*border));
            if(distance(c[id],c[current])>group_distance*module){current=id;c[g[0]].close.push_back(c[id].original);}
        }
    }
    std::vector<Candidate> result;for(int i=0;i<count;i++)if(selected[i])result.push_back(c[i]);
    for(int i=int(result.size())-1;i>=0;i--)for(int j=i-1;j>=0;j--)if(inside(result[i],result[j])) {
        result[i].parent=j;result[j].depth=std::max(result[j].depth,result[i].depth+1);break;
    }
    int offset=0;
    for(unsigned i=0;i<result.size();i++) {
        auto& c=result[i];int* m=metadata+5*i;m[0]=c.original;m[1]=c.parent;m[2]=c.depth;m[3]=offset;m[4]=c.close.size();
        for(int id:c.close)close_indices[offset++]=id;
    }return result.size();
}

__global__ void warp_kernel(const unsigned char* gray,int h,int w,const double* matrices,
                           const int* envs,unsigned char* out,int size,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*size*size)return;
    int q=i/(size*size),x=i%size,y=(i/size)%size;const double* m=matrices+q*9;
    double den=m[6]*x+m[7]*y+m[8],scale=den?1./den:0;
    int sx=__double2int_rn((m[0]*x+m[1]*y+m[2])*scale),sy=__double2int_rn((m[3]*x+m[4]*y+m[5])*scale);
    out[i]=(sx>=0&&sy>=0&&sx<w&&sy<h)?gray[(envs[q]*h+sy)*w+sx]:0;
}
extern "C" int ocv_warp(const unsigned char* gray,int h,int w,const double* matrices,const int* envs,
 unsigned char* out,int size,int count,cudaStream_t stream) {
    warp_kernel<<<(count*size*size+255)/256,256,0,stream>>>(gray,h,w,matrices,envs,out,size,count);return cudaGetLastError();
}

// Integer ROI copies retain original OpenCV cornerSubPix rather than changing
// its floating-point sampling/iteration rules in this first compatibility port.
__global__ void roi_kernel(const unsigned char* gray,int h,int w,const int* records,
                          unsigned char* out,int size,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*size*size)return;
    int k=i/(size*size),x=i%size,y=(i/size)%size;const int* r=records+3*k;
    out[i]=gray[(r[0]*h+max(0,min(h-1,r[2]+y)))*w+max(0,min(w-1,r[1]+x))];
}
extern "C" int ocv_rois(const unsigned char* gray,int h,int w,const int* records,
 unsigned char* out,int size,int count,cudaStream_t stream) {
    roi_kernel<<<(count*size*size+255)/256,256,0,stream>>>(gray,h,w,records,out,size,count);return cudaGetLastError();
}
