#ifdef __riscv_v
#include "../src/sobel.h"
#include <riscv_vector.h>
#include <cstdint>

static const int8_t KX[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
static const int8_t KY[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

static inline void sobel_scalar_pixel(const uint8_t* src,int16_t* Gx,int16_t* Gy,int w,int h,int y,int x){
  int sum_x=0,sum_y=0;
  for(int ky=-1;ky<=1;++ky){int py=y+ky;if(py<0||py>=h)continue;
    for(int kx=-1;kx<=1;++kx){int px=x+kx;if(px<0||px>=w)continue;
      uint8_t pix=src[py*w+px];sum_x+=pix*KX[ky+1][kx+1];sum_y+=pix*KY[ky+1][kx+1];
    }}
  Gx[y*w+x]=static_cast<int16_t>(sum_x);Gy[y*w+x]=static_cast<int16_t>(sum_y);
}

void sobel_rvv(const uint8_t* __restrict__ src,int16_t* __restrict__ Gx,int16_t* __restrict__ Gy,int w,int h){
  for(int y=0;y<h;++y){
    const int row_base=y*w;
    sobel_scalar_pixel(src,Gx,Gy,w,h,y,0);
    for(int x=1;x<w-1;){
      int remaining=(w-1)-x;if(remaining<=0)break;
      size_t vl=__riscv_vsetvl_e16m2((size_t)remaining);
      vint16m2_t acc_gx=__riscv_vmv_v_x_i16m2(0,vl);
      vint16m2_t acc_gy=__riscv_vmv_v_x_i16m2(0,vl);
      for(int ky=-1;ky<=1;++ky){
        int py=y+ky;if(py<0||py>=h)continue;
        const uint8_t* row_ptr=src+py*w;const int krow=ky+1;
        vuint8m1_t px_base=__riscv_vle8_v_u8m1(row_ptr+x-1,vl+2);
        vuint8m1_t px_left=px_base;
        vuint8m1_t px_center=__riscv_vslidedown_vx_u8m1(px_base,1,vl+2);
        vuint8m1_t px_right=__riscv_vslidedown_vx_u8m1(px_base,2,vl+2);
        vint16m2_t pxl=__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(px_left,vl));
        vint16m2_t pxc=__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(px_center,vl));
        vint16m2_t pxr=__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(px_right,vl));
        int8_t kxl=KX[krow][0],kxc=KX[krow][1],kxr=KX[krow][2];
        int8_t kyl=KY[krow][0],kyc=KY[krow][1],kyr=KY[krow][2];
        if(kxl!=0){vint16m2_t prod=__riscv_vmul_vx_i16m2(pxl,(int16_t)kxl,vl);acc_gx=__riscv_vadd_vv_i16m2(acc_gx,prod,vl);}
        if(kxc!=0){vint16m2_t prod=__riscv_vmul_vx_i16m2(pxc,(int16_t)kxc,vl);acc_gx=__riscv_vadd_vv_i16m2(acc_gx,prod,vl);}
        if(kxr!=0){vint16m2_t prod=__riscv_vmul_vx_i16m2(pxr,(int16_t)kxr,vl);acc_gx=__riscv_vadd_vv_i16m2(acc_gx,prod,vl);}
        if(kyl!=0){vint16m2_t prod=__riscv_vmul_vx_i16m2(pxl,(int16_t)kyl,vl);acc_gy=__riscv_vadd_vv_i16m2(acc_gy,prod,vl);}
        if(kyc!=0){vint16m2_t prod=__riscv_vmul_vx_i16m2(pxc,(int16_t)kyc,vl);acc_gy=__riscv_vadd_vv_i16m2(acc_gy,prod,vl);}
        if(kyr!=0){vint16m2_t prod=__riscv_vmul_vx_i16m2(pxr,(int16_t)kyr,vl);acc_gy=__riscv_vadd_vv_i16m2(acc_gy,prod,vl);}
      }
      __riscv_vse16_v_i16m2(Gx+row_base+x,acc_gx,vl);
      __riscv_vse16_v_i16m2(Gy+row_base+x,acc_gy,vl);
      x+=(int)vl;
    }
    if(w>1){sobel_scalar_pixel(src,Gx,Gy,w,h,y,w-1);}
  }
}

#endif
