// H4 planogram engine core — pure integer, no float. C reference for the
// 3-language benchmark. Bit-exact with the Python/Rust implementations.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define BIAS 512u
// H4 encode: (x,y,w,h) u8 -> (X,Y,Z,W) u16, 8 integer add/sub, unsigned
static inline void h4_encode(const uint8_t b[4], uint16_t e[4]){
    uint32_t x=b[0],y=b[1],w=b[2],h=b[3];
    e[0]=(uint16_t)(x+y+w+h);
    e[1]=(uint16_t)((x+w+BIAS)-(y+h));
    e[2]=(uint16_t)((x+y+BIAS)-(w+h));
    e[3]=(uint16_t)((x+h+BIAS)-(y+w));
}
static inline void h4_decode(const uint16_t e[4], uint8_t b[4]){
    uint32_t X=e[0],Y=e[1],Z=e[2],W=e[3];
    b[0]=(uint8_t)((X+Y+Z+W-3u*BIAS)>>2);
    b[1]=(uint8_t)(((X+Z+BIAS)-(Y+W))>>2);
    b[2]=(uint8_t)(((X+Y+BIAS)-(Z+W))>>2);
    b[3]=(uint8_t)(((X+W+BIAS)-(Y+Z))>>2);
}
static inline uint32_t h4_l1(const uint16_t a[4], const uint16_t b[4]){
    uint32_t s=0; for(int i=0;i<4;i++){ uint32_t hi=a[i]>b[i]?a[i]:b[i], lo=a[i]>b[i]?b[i]:a[i]; s+=hi-lo;} return s;
}

int main(int argc,char**argv){
    int N = argc>1? atoi(argv[1]) : 13184;
    uint8_t*  B = malloc(N*4);
    uint16_t* E = malloc(N*4*sizeof(uint16_t));
    srand(42);
    for(int i=0;i<N*4;i++) B[i]=(uint8_t)((i*167+13)&0xFF);
    // correctness: round-trip exact + checksum (bit-exactness across languages)
    int errs=0; uint64_t csum=0;
    for(int i=0;i<N;i++){ uint16_t e[4]; uint8_t d[4];
        h4_encode(&B[i*4],e); h4_decode(e,d);
        for(int j=0;j<4;j++){ if(d[j]!=B[i*4+j]) errs++; csum=csum*131+e[j]; E[i*4+j]=e[j]; } }
    // speed: encode timing
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    volatile uint64_t sink=0;
    for(int r=0;r<1000;r++) for(int i=0;i<N;i++){ uint16_t e[4]; h4_encode(&B[i*4],e); sink+=e[0]; }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ns=((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/(1000.0*N);
    printf("C   : N=%d  roundtrip_errors=%d  checksum=%llu  encode=%.4f ns/box\n",
           N,errs,(unsigned long long)csum,ns);
    free(B);free(E); return errs;
}
