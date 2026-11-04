/* rsqrtps of every float in [1, 4) (two binades: exponent parity), and a check that the result for any other
   normal input is the table entry of the same parity with the exponent rescaled (x*4^k -> r*2^-k). */
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
static float rsq1(float x){ __m128 v=_mm_set1_ps(x); v=_mm_rsqrt_ps(v); float r; _mm_store_ss(&r,v); return r; }
int main(int argc, char**argv){
  uint32_t n = 1u<<24; uint32_t *tab = malloc(n*4);
  for(uint32_t m=0;m<n;m+=4){ uint32_t b[4]; for(int k=0;k<4;k++) b[k]=0x3f800000u+(m+k);
    __m128 v; memcpy(&v,b,16); v=_mm_rsqrt_ps(v); memcpy(&tab[m],&v,16);}
  FILE*f=fopen(argv[1],"wb"); fwrite(tab,4,n,f); fclose(f);
  srand(1); long bad=0, tot=0;
  for(int t=0;t<20000000;t++){ uint32_t m=((uint32_t)rand()<<8 ^ rand()) & 0x7fffff; int e=(rand()%200)+27; uint32_t xb=((uint32_t)e<<23)|m;
    float x; memcpy(&x,&xb,4); float r=rsq1(x); uint32_t rb; memcpy(&rb,&r,4);
    int par=(e-127)&1; int k=((e-127)-par)/2;   /* x = y * 4^k, y in [1,4) */
    uint32_t tb=tab[((uint32_t)par<<23)|m]; int te=(int)((tb>>23)&0xff) - k; uint32_t pred=(tb&0x807fffffu)|((uint32_t)te<<23);
    if(te>0 && te<255){ tot++; if(pred!=rb) bad++; } }
  printf("checked %ld, mismatches %ld\n", tot, bad);
  return 0; }
