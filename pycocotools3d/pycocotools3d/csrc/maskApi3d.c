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

#include "maskApi3d.h"
#include <math.h>
#include <stdlib.h>

uint umin( uint a, uint b ) { return (a<b) ? a : b; }
uint umax( uint a, uint b ) { return (a>b) ? a : b; }

void rleInit3D( RLE3D *R, siz d, siz h, siz w, siz m, uint *cnts ) {
  R->d=d; R->h=h; R->w=w; R->m=m; 
  R->cnts=(m==0)?0:malloc(sizeof(uint)*m);
  siz j; 
  if(cnts) 
    for(j=0; j<m; j++) 
      R->cnts[j]=cnts[j];
}

void rleFree3D( RLE3D *R ) {
  free(R->cnts); R->cnts=0;
}

void rlesInit3D( RLE3D **R, siz n ) {
  siz i; *R = (RLE3D*) malloc(sizeof(RLE3D)*n);
  for(i=0; i<n; i++) 
    rleInit3D((*R)+i,0,0,0,0,0);
}

void rlesFree3D( RLE3D **R, siz n ) {
  siz i; 
  for(i=0; i<n; i++) 
    rleFree3D((*R)+i);
  free(*R); *R=0;
}

void rleEncode3D( RLE3D *R, const byte *M, siz d, siz h, siz w, siz n ) {
  siz i, j, k, a=w*h*d; uint c, *cnts; byte p;
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
    cnts[k++]=c; 
    rleInit3D(R+i,d,h,w,k,cnts);
  }
  free(cnts);
}

// Decodes n different RLE3Ds that have the same width and height. Write results to M.
// Returns whether the decoding succeeds or not.
byte rleDecode3D( const RLE3D *R, byte *M, siz n ) {
  // Safeguards for memory boundary
  siz s=R[0].d*R[0].h*R[0].w*n;
  siz c=0;
  siz i, j, k; 
  for( i=0; i<n; i++ ) {
    byte v=0; 
    for( j=0; j<R[i].m; j++ ) {
      for( k=0; k<R[i].cnts[j]; k++ ) {
        if ( c >= s ) {
          // Memory boundary would be crossed, wrong RLE3D
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

void rleMerge3D( const RLE3D *R, RLE3D *M, siz n, int intersect ) {
  uint *cnts, c, ca, cb, cc, ct; 
  int v, va, vb, vp;
  siz i, a, b, d=R[0].d, h=R[0].h, w=R[0].w, m=R[0].m; RLE3D A, B;
  if(n==0) { rleInit3D(M,0,0,0,0,0); return; }
  if(n==1) { rleInit3D(M,d,h,w,m,R[0].cnts); return; }
  cnts = malloc(sizeof(uint)*(d*h*w+1));
  for( a=0; a<m; a++ ) cnts[a]=R[0].cnts[a];
  for( i=1; i<n; i++ ) {
    B=R[i]; 
    if(B.d!=d||B.h!=h||B.w!=w) { 
        h=w=m=0; 
        break; 
    }
    rleInit3D(&A,d,h,w,m,cnts); 
    ca=A.cnts[0]; 
    cb=B.cnts[0];
    v=va=vb=0; m=0; a=b=1; cc=0; ct=1;
    while( ct>0 ) {
      c=umin(ca,cb); cc+=c; ct=0;
      ca-=c; if(!ca && a<A.m) { ca=A.cnts[a++]; va=!va; } ct+=ca;
      cb-=c; if(!cb && b<B.m) { cb=B.cnts[b++]; vb=!vb; } ct+=cb;
      vp=v; if(intersect) v=va&&vb; else v=va||vb;
      if( v!=vp||ct==0 ) { 
          cnts[m++]=cc; 
          cc=0; 
      }
    }
    rleFree3D(&A);
  }
  rleInit3D(M,d,h,w,m,cnts); 
  free(cnts);
}

void rleArea3D( const RLE3D *R, siz n, uint *a ) {
  siz i, j; 
  for( i=0; i<n; i++ ) {
    a[i]=0; 
    for( j=1; j<R[i].m; j+=2 ) 
      a[i]+=R[i].cnts[j];
  }
}

void rleIou3D( RLE3D *dt, RLE3D *gt, siz m, siz n, byte *iscrowd, double *o ) {
  siz g, d; 
  BB3D db, gb; 
  int crowd;
  db=malloc(sizeof(double)*m*6); rleToBbox3D(dt,db,m);
  gb=malloc(sizeof(double)*n*6); rleToBbox3D(gt,gb,n);
  bbIou3D(db,gb,m,n,iscrowd,o); 
  free(db); free(gb);
  for( g=0; g<n; g++ ) 
    for( d=0; d<m; d++ ) 
      if(o[g*m+d]>0) {
        crowd=iscrowd!=NULL && iscrowd[g];
        if(dt[d].d!=gt[g].d || dt[d].h!=gt[g].h || dt[d].w!=gt[g].w) { 
            o[g*m+d]=-1; 
            continue; 
        }
        siz ka, kb, a, b; 
        uint c, ca, cb, ct, i, u; 
        int va, vb;
        ca=dt[d].cnts[0]; ka=dt[d].m;
        cb=gt[g].cnts[0]; kb=gt[g].m; 
        va=vb=0;
        a=b=1; i=u=0; ct=1;
        while( ct>0 ) {
            c=umin(ca,cb); if(va||vb) { u+=c; if(va&&vb) i+=c; } ct=0;
            ca-=c; if(!ca && a<ka) { ca=dt[d].cnts[a++]; va=!va; } ct+=ca;
            cb-=c; if(!cb && b<kb) { cb=gt[g].cnts[b++]; vb=!vb; } ct+=cb;
        }
        if(i==0) u=1; else if(crowd) rleArea3D(dt+d,1,&u);
        o[g*m+d] = (double)i/(double)u;
  }
}

void rleNms3D( RLE3D *dt, siz n, uint *keep, double thr ) {
  siz i, j; double u;
  for( i=0; i<n; i++ ) keep[i]=1;
  for( i=0; i<n; i++ ) if(keep[i]) {
    for( j=i+1; j<n; j++ ) if(keep[j]) {
      rleIou3D(dt+i,dt+j,1,1,0,&u);
      if(u>thr) keep[j]=0;
    }
  }
}

void bbIou3D( BB3D dt, BB3D gt, siz m, siz n, byte *iscrowd, double *o ) {
  double d, h, w, i, u, ga, da; siz g, d_; int crowd;
  for( g=0; g<n; g++ ) {
    BB3D G=gt+g*6; 
    ga=G[3]*G[4]*G[5]; 
    crowd=iscrowd!=NULL && iscrowd[g];
    for( d_=0; d_<m; d_++ ) {
      BB3D D=dt+d_*6; 
      da=D[3]*D[4]*D[5]; 
      o[g*m+d_]=0;
      d=fmin(D[3]+D[0],G[3]+G[0])-fmax(D[0],G[0]); if(d<=0) continue;
      h=fmin(D[4]+D[1],G[4]+G[1])-fmax(D[1],G[1]); if(h<=0) continue;
      w=fmin(D[5]+D[2],G[5]+G[2])-fmax(D[2],G[2]); if(w<=0) continue;
      i=w*h*d; 
      u=crowd ? da : da+ga-i;
      o[g*m+d_]=i/u;
    }
  }
}

void bbNms3D( BB3D dt, siz n, uint *keep, double thr ) {
  siz i, j; double u;
  for( i=0; i<n; i++ ) keep[i]=1;

  for( i=0; i<n; i++ ) 
    if(keep[i])
      for( j=i+1; j<n; j++ ) 
        if(keep[j]) {
          bbIou3D(dt+i*6,dt+j*6,1,1,0,&u);
          if(u>thr) keep[j]=0;
        }
}

void rleToBbox3D( const RLE3D *R, BB3D bb, siz n ) {
  siz i; 
  for( i=0; i<n; i++ ) {
    uint d, h, w, xs, ys, zs, xe, ye, ze, cc; siz j, m;
    d=(uint)R[i].d; h=(uint)R[i].h; w=(uint)R[i].w; 
    m=R[i].m;
    m=((siz)(m/2))*2; 
    xs=w; ys=h, zs=d; xe=ye=ze=0; cc=0;

    if(m==0) { bb[6*i+0]=bb[6*i+1]=bb[6*i+2]=bb[6*i+3]=bb[6*i+4]=bb[6*i+5]=0; continue; }
    for( j=0; j<m; j++ ) {
      uint start = cc;   // start of current segment
      cc+=R[i].cnts[j];  // start of next segment
      if (j % 2 == 0) continue; // skip background segment
      if (R[i].cnts[j] == 0) continue; // skip zero-length foreground segment

      // Fortran ordered
      uint x_start = start / (h * d), y_start = (start / d) % h, z_start = start % d;
      uint x_end = (cc - 1) / (h * d), y_end = ((cc - 1) / d) % h, z_end = (cc - 1) % d;

      xs = umin(xs, x_start);
      xe = umax(xe, x_end);
      ys = umin(ys, y_start);
      ye = umax(ye, y_end);
      zs = umin(zs, z_start);
      ze = umax(ze, z_end);

      // Detect segment going across depth
      if (R[i].cnts[j] > d - z_start) {
        zs = 0; ze = d - 1;
      }
      // Detect segment going across height
      if (R[i].cnts[j] > d - z_start + d * (h - y_start - 1)) {
        ys = 0; ye = h - 1;
      }
    }

    bb[6*i]  =zs; bb[6*i+3]=ze-zs+1;
    bb[6*i+1]=ys; bb[6*i+4]=ye-ys+1;
    bb[6*i+2]=xs; bb[6*i+5]=xe-xs+1;
  }
}

void rleFrBbox3D( RLE3D *R, const BB3D bb, siz d, siz h, siz w, siz n ) {
  siz i, x, y, z, a=w*h*d; byte *M;
  M = malloc(sizeof(byte) * a * n);
  // Convert bounding box to explicit mask and then encode using rleEncode3D
  for( i=0; i<n; i++ ) {
    double zs=bb[6*i],   ze=zs+bb[6*i+3];
    double ys=bb[6*i+1], ye=ys+bb[6*i+4];
    double xs=bb[6*i+2], xe=xs+bb[6*i+5];
    byte *T=M+a*i;
    for (x=0; x<w; x++)
      for (y=0; y<h; y++)
        for (z=0; z<d; z++)
          T[x * h * d + y * d + z] = (xs <= x && x < xe && ys <=y && y < ye && zs <= z && z < ze) ? 1 : 0;
  }
  rleEncode3D( R, M, d, h, w, n );
}

char* rleToString3D( const RLE3D *R ) {
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

void rleFrString3D( RLE3D *R, char *s, siz d, siz h, siz w ) {
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
  rleInit3D(R,d,h,w,m,cnts);
  free(cnts);
}
