#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
static float rcp1(float x){ __m128 v=_mm_set1_ps(x); v=_mm_rcp_ps(v); float r; _mm_store_ss(&r,v); return r; }
int main(int argc, char**argv){
  uint32_t n = 1u<<23; uint32_t *tab = malloc(n*4);
  for(uint32_t m=0;m<n;m+=4){ uint32_t b[4]; for(int k=0;k<4;k++) b[k]=0x3f800000u|(m+k);
    __m128 v; memcpy(&v,b,16); v=_mm_rcp_ps(v); memcpy(&tab[m],&v,16);}
  FILE*f=fopen(argv[1],"wb"); fwrite(tab,4,n,f); fclose(f);
  /* exponent independence check on random normal floats */
  srand(1); long bad=0, tot=0;
  for(int t=0;t<20000000;t++){ uint32_t m=((uint32_t)rand()<<8 ^ rand()) & 0x7fffff; int e=(rand()%200)+27; uint32_t xb=((uint32_t)e<<23)|m;
    float x; memcpy(&x,&xb,4); float r=rcp1(x); uint32_t rb; memcpy(&rb,&r,4);
    uint32_t tb=tab[m]; int te=(int)((tb>>23)&0xff) - (e-127); uint32_t pred=(tb&0x807fffffu)|((uint32_t)te<<23);
    if(te>0 && te<255){ tot++; if(pred!=rb) bad++; } }
  printf("checked %ld, mismatches %ld\n", tot, bad);
  return 0; }
