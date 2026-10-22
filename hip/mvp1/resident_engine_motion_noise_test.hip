// Asset-free integration: same input + stationary valid guides must retain
// noise exactly at history0, including later passes. Missing/depth-changing
// guides must refresh it. Config needs noise_motion>0 and external_controls=1.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
static int checks=0;
static void check(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);++checks;printf("PASS %s\n",message.c_str());}
static void hc(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
int main(int argc,char** argv){
 if(argc!=3){fputs("usage: resident_engine_motion_noise_test engine.dll motion_config\n",stderr);return 2;}
 HMODULE dll=LoadLibraryA(argv[1]);if(!dll){fputs("cannot load engine\n",stderr);return 2;}
 auto create=(int(*)(const char*))GetProcAddress(dll,"re_create");auto destroy=(void(*)())GetProcAddress(dll,"re_destroy");
 auto passes=(int(*)(int))GetProcAddress(dll,"re_set_passes");auto history=(int(*)(float))GetProcAddress(dll,"re_set_history");
 auto controls=(int(*)(const float*,int))GetProcAddress(dll,"re_controls");auto motion=(int(*)(const void*,int,int,int))GetProcAddress(dll,"re_motion");
 auto depth=(int(*)(const void*,int,int,int))GetProcAddress(dll,"re_depth");auto submit=(int(*)(const void*,int,int,int,int,int))GetProcAddress(dll,"re_submit_full");
 auto completed=(unsigned long long(*)())GetProcAddress(dll,"re_completed");auto compose=(int(*)(void*,unsigned long long,int,int,int,float*))GetProcAddress(dll,"re_compose_device");
 auto error=(const char*(*)())GetProcAddress(dll,"re_last_error");
 void *in=nullptr,*out=nullptr,*mv=nullptr,*z=nullptr;int result=1;
 try{
  if(!create||!destroy||!passes||!history||!controls||!motion||!depth||!submit||!completed||!compose||!error)throw std::runtime_error("missing export");
  constexpr int w=320,h=320,n=w*h;
  std::vector<_Float16> frame(n*4);std::vector<float> zs(n,0.5f);
  for(int y=0;y<h;++y)for(int x=0;x<w;++x){int p=y*w+x;frame[p*4]=(_Float16)(0.15f+0.45f*x/w+(((x/16+y/16)&1)?0.12f:0));frame[p*4+1]=(_Float16)(0.2f+0.4f*y/h);frame[p*4+2]=(_Float16)0.45f;frame[p*4+3]=(_Float16)1;}
  hc(hipMalloc(&in,n*8));hc(hipMalloc(&out,n*8));hc(hipMalloc(&mv,n*8));hc(hipMalloc(&z,n*4));hc(hipMemcpy(in,frame.data(),n*8,hipMemcpyHostToDevice));hc(hipMemset(mv,0,n*8));
  const float settings[6]={1,0,1,1,0.73f,1};
  auto run=[&](int p,bool guides){
   if(create(argv[2])!=1||passes(p)!=1||history(0)!=1||controls(settings,6)!=1)throw std::runtime_error(error());
   std::vector<std::vector<_Float16>> r;
   for(int f=0;f<4;++f){
    hc(hipMemcpy(out,frame.data(),n*8,hipMemcpyHostToDevice));
    if(motion(mv,w,h,w*8)!=1)throw std::runtime_error(error());
    if(guides){std::fill(zs.begin(),zs.end(),f<3?0.5f:0.25f);hc(hipMemcpy(z,zs.data(),n*4,hipMemcpyHostToDevice));if(depth(z,w,h,w*4)!=1)throw std::runtime_error(error());}
    auto before=completed();if(submit(in,w,h,w*8,8,0)!=1)throw std::runtime_error(error());auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(40);
    while(completed()==before){if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error(error());std::this_thread::sleep_for(std::chrono::milliseconds(1));}
    float ms=0;if(compose(out,w*8,w,h,8,&ms)!=1)throw std::runtime_error(error());r.emplace_back(n*4);hc(hipMemcpy(r.back().data(),out,n*8,hipMemcpyDeviceToHost));
   }
   destroy();return r;
  };
  for(int p=1;p<=3;++p){
   auto a=run(p,true);bool finite=true;for(auto& f:a)for(size_t i=0;i<f.size();++i)finite &= std::isfinite(float(f[i]));
   check(finite,"finite motion-noise output at passes"+std::to_string(p));
   check(a[1]==a[2],"stationary valid guides preserve warm output at passes"+std::to_string(p));
   check(a[2]!=a[3],"depth change refreshes noise at passes"+std::to_string(p));
   auto b=run(p,false);check(b[1]!=b[2],"missing depth refreshes noise at passes"+std::to_string(p));
  }
  printf("motion noise integration: %d/%d PASS\n",checks,checks);result=0;
 }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());}
 if(destroy)destroy();if(in)hipFree(in);if(out)hipFree(out);if(mv)hipFree(mv);if(z)hipFree(z);FreeLibrary(dll);return result;
}
