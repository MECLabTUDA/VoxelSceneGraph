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
#pragma once

typedef unsigned int uint;
typedef unsigned long siz;
typedef unsigned char byte;
typedef double* BB3D;
typedef struct { siz d, h, w, m; uint *cnts; } RLE3D;

/* Initialize/destroy RLE3D. */
void rleInit3D( RLE3D *R, siz d, siz h, siz w, siz m, uint *cnts );
void rleFree3D( RLE3D *R );

/* Initialize/destroy RLE3D array. */
void rlesInit3D( RLE3D **R, siz n );
void rlesFree3D( RLE3D **R, siz n );

/* Encode binary masks using RLE. */
void rleEncode3D( RLE3D *R, const byte *mask, siz d, siz h, siz w, siz n );

/* Decode binary masks encoded via RLE. */
byte rleDecode3D( const RLE3D *R, byte *mask, siz n );

/* Compute union or intersection of encoded masks. */
void rleMerge3D( const RLE3D *R, RLE3D *M, siz n, int intersect );

/* Compute area of encoded masks. */
void rleArea3D( const RLE3D *R, siz n, uint *a );

/* Compute intersection over union between masks. */
void rleIou3D( RLE3D *dt, RLE3D *gt, siz m, siz n, byte *iscrowd, double *o );

/* Compute non-maximum suppression between bounding masks */
void rleNms3D( RLE3D *dt, siz n, uint *keep, double thr );

/* Compute intersection over union between bounding boxes. */
void bbIou3D( BB3D dt, BB3D gt, siz m, siz n, byte *iscrowd, double *o );

/* Compute non-maximum suppression between bounding boxes */
void bbNms3D( BB3D dt, siz n, uint *keep, double thr );

/* Get bounding boxes surrounding encoded masks. */
void rleToBbox3D( const RLE3D *R, BB3D bb, siz n );

/* Convert bounding boxes to encoded masks. */
void rleFrBbox3D( RLE3D *R, const BB3D bb, siz d, siz h, siz w, siz n );

/* Get compressed string representation of encoded mask. */
char* rleToString3D( const RLE3D *R );

/* Convert from compressed string representation of encoded mask. */
void rleFrString3D( RLE3D *R, char *s, siz d, siz h, siz w );
