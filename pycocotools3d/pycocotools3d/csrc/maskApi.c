/*
Copyright 2023 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
#include "maskApi.h"
#include <math.h>
#include <stdlib.h>

uint umin( uint a, uint b ) { return (a<b) ? a : b; }
uint umax( uint a, uint b ) { return (a>b) ? a : b; }

void rleInit( RLE *R, siz h, siz w, siz m, uint *cnts ) {
  R->h=h; R->w=w; R->m=m; R->cnts=(m==0)?0:malloc(sizeof(uint)*m);
  siz j; if(cnts) for(j=0; j<m; j++) R->cnts[j]=cnts[j];
}

void rleFree( RLE *R ) {
  free(R->cnts); R->cnts=0;
}

void rlesInit( RLE **R, siz n ) {
  siz i; *R = (RLE*) malloc(sizeof(RLE)*n);
  for(i=0; i<n; i++) rleInit((*R)+i,0,0,0,0);
}

void rlesFree( RLE **R, siz n ) {
  siz i; for(i=0; i<n; i++) rleFree((*R)+i); free(*R); *R=0;
}

void rleEncode( RLE *R, const byte *M, siz h, siz w, siz n ) {
  siz i, j, k, a=w*h; uint c, *cnts; byte p;
  cnts = malloc(sizeof(uint)*(a+1));
  for(i=0; i<n; i++) {
    const byte *T=M+a*i; k=0; p=0; c=0;
    for(j=0; j<a; j++) {
      if(T[j]!=p) {
        cnts[k++]=c;
        c=0; p=T[j];
        }
      c++;
    }
    cnts[k++]=c; rleInit(R+i,h,w,k,cnts);
  }
  free(cnts);
}

// Decodes n different RLEs that have the same width and height. Write results to M.
// Returns whether the decoding succeeds or not.
byte rleDecode( const RLE *R, byte *M, siz n ) {
  // Safeguards for memory boundary
  siz s=R[0].h*R[0].w*n;
  siz c=0;
  siz i, j, k; for( i=0; i<n; i++ ) {
    byte v=0; for( j=0; j<R[i].m; j++ ) {
      for( k=0; k<R[i].cnts[j]; k++ ) {
        if ( c >= s ) {
          // Memory boundary would be crossed, wrong RLE
          return 0;
        }
        c++;
        *(M++)=v;
      }
      v=!v;
    }
  }
  return 1;
}

void rleMerge( const RLE *R, RLE *M, siz n, int intersect ) {
  uint *cnts, c, ca, cb, cc, ct; int v, va, vb, vp;
  siz i, a, b, h=R[0].h, w=R[0].w, m=R[0].m; RLE A, B;
  if(n==0) { rleInit(M,0,0,0,0); return; }
  if(n==1) { rleInit(M,h,w,m,R[0].cnts); return; }
  cnts = malloc(sizeof(uint)*(h*w+1));
  for( a=0; a<m; a++ ) cnts[a]=R[0].cnts[a];
  for( i=1; i<n; i++ ) {
    B=R[i]; if(B.h!=h||B.w!=w) { h=w=m=0; break; }
    rleInit(&A,h,w,m,cnts); ca=A.cnts[0]; cb=B.cnts[0];
    v=va=vb=0; m=0; a=b=1; cc=0; ct=1;
    while( ct>0 ) {
      c=umin(ca,cb); cc+=c; ct=0;
      ca-=c; if(!ca && a<A.m) { ca=A.cnts[a++]; va=!va; } ct+=ca;
      cb-=c; if(!cb && b<B.m) { cb=B.cnts[b++]; vb=!vb; } ct+=cb;
      vp=v; if(intersect) v=va&&vb; else v=va||vb;
      if( v!=vp||ct==0 ) { cnts[m++]=cc; cc=0; }
    }
    rleFree(&A);
  }
  rleInit(M,h,w,m,cnts); free(cnts);
}

void rleArea( const RLE *R, siz n, uint *a ) {
  siz i, j; for( i=0; i<n; i++ ) {
    a[i]=0; for( j=1; j<R[i].m; j+=2 ) a[i]+=R[i].cnts[j]; }
}

void rleIou( RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o ) {
  siz g, d; BB db, gb; int crowd;
  db=malloc(sizeof(double)*m*4); rleToBbox(dt,db,m);
  gb=malloc(sizeof(double)*n*4); rleToBbox(gt,gb,n);
  bbIou(db,gb,m,n,iscrowd,o); free(db); free(gb);
  for( g=0; g<n; g++ ) for( d=0; d<m; d++ ) if(o[g*m+d]>0) {
    crowd=iscrowd!=NULL && iscrowd[g];
    if(dt[d].h!=gt[g].h || dt[d].w!=gt[g].w) { o[g*m+d]=-1; continue; }
    siz ka, kb, a, b; uint c, ca, cb, ct, i, u; int va, vb;
    ca=dt[d].cnts[0]; ka=dt[d].m; va=vb=0;
    cb=gt[g].cnts[0]; kb=gt[g].m; a=b=1; i=u=0; ct=1;
    while( ct>0 ) {
      c=umin(ca,cb); if(va||vb) { u+=c; if(va&&vb) i+=c; } ct=0;
      ca-=c; if(!ca && a<ka) { ca=dt[d].cnts[a++]; va=!va; } ct+=ca;
      cb-=c; if(!cb && b<kb) { cb=gt[g].cnts[b++]; vb=!vb; } ct+=cb;
    }
    if(i==0) u=1; else if(crowd) rleArea(dt+d,1,&u);
    o[g*m+d] = (double)i/(double)u;
  }
}

void rleNms( RLE *dt, siz n, uint *keep, double thr ) {
  siz i, j; double u;
  for( i=0; i<n; i++ ) keep[i]=1;
  for( i=0; i<n; i++ ) if(keep[i]) {
    for( j=i+1; j<n; j++ ) if(keep[j]) {
      rleIou(dt+i,dt+j,1,1,0,&u);
      if(u>thr) keep[j]=0;
    }
  }
}

void bbIou( BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o ) {
  double h, w, i, u, ga, da; siz g, d; int crowd;
  for( g=0; g<n; g++ ) {
    BB G=gt+g*4; ga=G[2]*G[3]; crowd=iscrowd!=NULL && iscrowd[g];
    for( d=0; d<m; d++ ) {
      BB D=dt+d*4; da=D[2]*D[3]; o[g*m+d]=0;
      h=fmin(D[2]+D[0],G[2]+G[0])-fmax(D[0],G[0]); if(h<=0) continue;
      w=fmin(D[3]+D[1],G[3]+G[1])-fmax(D[1],G[1]); if(w<=0) continue;
      i=w*h; u = crowd ? da : da+ga-i; o[g*m+d]=i/u;
    }
  }
}

void bbNms( BB dt, siz n, uint *keep, double thr ) {
  siz i, j; double u;
  for( i=0; i<n; i++ ) keep[i]=1;
  for( i=0; i<n; i++ ) if(keep[i]) {
    for( j=i+1; j<n; j++ ) if(keep[j]) {
      bbIou(dt+i*4,dt+j*4,1,1,0,&u);
      if(u>thr) keep[j]=0;
    }
  }
}

void rleToBbox( const RLE *R, BB bb, siz n ) {
  siz i; for( i=0; i<n; i++ ) {
    uint h, w, xs, ys, xe, ye, cc; siz j, m;
    h=(uint)R[i].h; w=(uint)R[i].w; m=R[i].m;
    m=((siz)(m/2))*2; xs=w; ys=h; xe=ye=0; cc=0;
    if(m==0) { bb[4*i]=bb[4*i+1]=bb[4*i+2]=bb[4*i+3]=0; continue; }
    for( j=0; j<m; j++ ) {
      uint start = cc;   // start of current segment
      cc+=R[i].cnts[j];  // start of next segment
      if (j % 2 == 0) continue; // skip background segment
      if (R[i].cnts[j] == 0) continue; // skip zero-length foreground segment
      uint y_start = start % h, x_start = (start - y_start) / h;
      uint y_end = (cc - 1) % h, x_end = (cc - 1 - y_end) / h;

      // x_start <= x_end must be true
      xs = umin(xs, x_start);
      xe = umax(xe, x_end);

      if (x_start < x_end) {
        ys = 0; ye = h - 1;    // foreground segment goes across columns
      } else {
        // if x_start == x_end, then y_start <= y_end must be true
        ys = umin(ys, y_start);
        ye = umax(ye, y_end);
      }
    }
    bb[4*i]  =ys; bb[4*i+2]=ye-ys+1;
    bb[4*i+1]=xs; bb[4*i+3]=xe-xs+1;
  }
}

void rleFrBbox( RLE *R, const BB bb, siz h, siz w, siz n ) {
  siz i; for( i=0; i<n; i++ ) {
    double ys=bb[4*i],   ye=ys+bb[4*i+2];
    double xs=bb[4*i+1], xe=xs+bb[4*i+3];
    double yx[8] = {ys,xs,ys,xe,ye,xe,ye,xs};
    rleFrPoly( R+i, yx, 4, h, w );
  }
}

int uintCompare(const void *a, const void *b) {
  uint c=*((uint*)a), d=*((uint*)b); return c>d?1:c<d?-1:0;
}

void rleFrPoly( RLE *R, const double *yx, siz k, siz h, siz w ) {
  /* upsample and get discrete points densely along entire boundary */
  siz j, m=0; double scale=5; int *x, *y, *u, *v; uint *a, *b;
  x=malloc(sizeof(int)*(k+1)); y=malloc(sizeof(int)*(k+1));
  for(j=0; j<k; j++) y[j]=(int)(scale*yx[j*2]+.5);
  y[k]=y[0];
  for(j=0; j<k; j++) x[j]=(int)(scale*yx[j*2+1]+.5);
  x[k]=x[0];
  for(j=0; j<k; j++) m+=umax(abs(x[j]-x[j+1]), abs(y[j]-y[j+1])) + 1;
  u=malloc(sizeof(int)*m); v=malloc(sizeof(int)*m); m=0;
  for( j=0; j<k; j++ ) {
    int xs=x[j], xe=x[j+1], ys=y[j], ye=y[j+1], dx, dy, t, d;
    int flip; double s; dx=abs(xe-xs); dy=abs(ys-ye);
    flip = (dx>=dy && xs>xe) || (dx<dy && ys>ye);
    if(flip) { t=xs; xs=xe; xe=t; t=ys; ys=ye; ye=t; }
    s = dx>=dy ? (double)(ye-ys)/dx : (double)(xe-xs)/dy;
    if(dx>=dy) for( d=0; d<=dx; d++ ) {
      t=flip?dx-d:d; u[m]=t+xs; v[m]=(int)(ys+s*t+.5); m++;
    } else for( d=0; d<=dy; d++ ) {
      t=flip?dy-d:d; v[m]=t+ys; u[m]=(int)(xs+s*t+.5); m++;
    }
  }
  /* get points along y-boundary and downsample */
  free(x); free(y); k=m; m=0; double xd, yd;
  x=malloc(sizeof(int)*k); y=malloc(sizeof(int)*k);
  for( j=1; j<k; j++ ) if(u[j]!=u[j-1]) {
    xd=(double)(u[j]<u[j-1]?u[j]:u[j]-1); xd=(xd+.5)/scale-.5;
    if( floor(xd)!=xd || xd<0 || xd>w-1 ) continue;
    yd=(double)(v[j]<v[j-1]?v[j]:v[j-1]); yd=(yd+.5)/scale-.5;
    if(yd<0) yd=0; else if(yd>h) yd=h; yd=ceil(yd);
    x[m]=(int) xd; y[m]=(int) yd; m++;
  }
  /* compute rle encoding given y-boundary points */
  k=m; a=malloc(sizeof(uint)*(k+1));
  for( j=0; j<k; j++ ) a[j]=(uint)(x[j]*(int)(h)+y[j]);
  a[k++]=(uint)(h*w); free(u); free(v); free(x); free(y);
  qsort(a,k,sizeof(uint),uintCompare); uint p=0;
  for( j=0; j<k; j++ ) { uint t=a[j]; a[j]-=p; p=t; }
  b=malloc(sizeof(uint)*k); j=m=0; b[m++]=a[j++];
  while(j<k) if(a[j]>0) b[m++]=a[j++]; else {
    j++; if(j<k) b[m-1]+=a[j++]; }
  rleInit(R,h,w,m,b); free(a); free(b);
}

char* rleToString( const RLE *R ) {
  /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
  siz i, m=R->m, p=0; long x; int more;
  char *s=malloc(sizeof(char)*m*6);
  for( i=0; i<m; i++ ) {
    x=(long) R->cnts[i]; if(i>2) x-=(long) R->cnts[i-2]; more=1;
    while( more ) {
      char c=x & 0x1f; x >>= 5; more=(c & 0x10) ? x!=-1 : x!=0;
      if(more) c |= 0x20;
      c+=48; s[p++]=c;
    }
  }
  s[p]=0; return s;
}

void rleFrString( RLE *R, char *s, siz h, siz w ) {
  siz m=0, p=0, k; long x; int more; uint *cnts;
  while( s[m] ) m++;
  cnts=malloc(sizeof(uint)*m); m=0;
  while( s[p] ) {
    x=0; k=0; more=1;
    while( more ) {
      char c=s[p]-48; x |= (c & 0x1f) << 5*k;
      more = c & 0x20; p++; k++;
      if(!more && (c & 0x10)) x |= -1 << 5*k;
    }
    if(m>2) x+=(long) cnts[m-2];
    cnts[m++]=(uint) x;
  }
  rleInit(R,h,w,m,cnts); free(cnts);
}
