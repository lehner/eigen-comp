#define _FILE_OFFSET_BITS 64
#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex>
#include <vector>
#include <memory.h>

using namespace std;

int nthreads;
uint32_t crc32_fast(const void* data, size_t length, uint32_t previousCrc32);

inline double dclock() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (1.0*tv.tv_usec + 1.0e6*tv.tv_sec) / 1.0e6;
}

const char* header = 
  "QCD eigenvector decompressor\n"
  "Authors: Christoph Lehner\n"
  "Date: 2017\n";

struct _evc_meta_ {
  int s[5];
  int b[5];
  int nkeep;
  int nkeep_single;

  // derived
  int nb[5];
  int blocks;

  int neig;

  int index;

  uint32_t crc32;

  int FP16_COEF_EXP_SHARE_FLOATS;
};

int vol4d, vol5d;
int f_size, f_size_block, f_size_coef_block, nkeep_fp16;

char* raw_in;

#ifndef OPT
//#define OPT double
#define OPT float
#endif

vector< vector<OPT> > block_data;
vector< vector<OPT> > block_data_ortho;
vector< vector<OPT> > block_coef;

//float* ord_in; // in non-bfm ordering:  co fastest, then x,y,z,t,s

void fix_float_endian(float* dest, int nfloats) {
  int n_endian_test = 1;
  bool machine_is_little_endian = *(char *)&n_endian_test == 1;
  int bigendian = 1; // write data back in big endian

  if ((bigendian && machine_is_little_endian) ||  // written for readability
      (!bigendian && !machine_is_little_endian)) {
    int i;
    for (i=0;i<nfloats;i++) {
      char* c = (char*)&dest[i];
      char tmp;
      int j;
      for (j=0;j<2;j++) {
	tmp = c[j];
	c[j] = c[3-j];
	c[3-j] = tmp;
      }
    }	
  }

}

_evc_meta_ args;

int get_bfm_index( int* pos, int co ) {

  int ls = args.s[4];
  int vol_4d_oo = vol4d / 2;
  int vol_5d = vol_4d_oo * ls;
  
  int NtHalf = args.s[3] / 2;
  int simd_coor = pos[3] / NtHalf;
  int regu_coor = (pos[0] + args.s[0] * (pos[1] + args.s[1] * ( pos[2] + args.s[2] * (pos[3] % NtHalf) ) )) / 2;
  int regu_vol  = vol_4d_oo / 2;

  return
    + regu_coor * ls * 48
    + pos[4] * 48
    + co * 4  
    + simd_coor * 2;
}

void index_to_pos(int i, int* pos, int* latt) {
  int d;
  for (d=0;d<5;d++) {
    pos[d] = i % latt[d];
    i /= latt[d];
  }
}

int pos_to_index(int* pos, int* latt) {
  return pos[0] + latt[0]*( pos[1] + latt[1]*( pos[2] + latt[2]*( pos[3] + latt[3]*pos[4] ) ) );
}


void pos_to_blocked_pos(int* pos, int* pos_in_block, int* block_coor) {
  int d;
  for (d=0;d<5;d++) {
    block_coor[d] = pos[d] / args.b[d];
    pos_in_block[d] = pos[d] - block_coor[d] * args.b[d];
  }
}

template<class T>
void caxpy_single(T* res, complex<T> ca, T* x, T* y, int f_size) {
  complex<T>* cx = (complex<T>*)x;
  complex<T>* cy = (complex<T>*)y;
  complex<T>* cres = (complex<T>*)res;
  int c_size = f_size / 2;

  for (int i=0;i<c_size;i++)
    cres[i] = ca*cx[i] + cy[i];
}    

template<class T>
void caxpy_threaded(T* res, complex<T> ca, T* x, T* y, int f_size) {
  complex<T>* cx = (complex<T>*)x;
  complex<T>* cy = (complex<T>*)y;
  complex<T>* cres = (complex<T>*)res;
  int c_size = f_size / 2;

#pragma omp for
  for (int i=0;i<c_size;i++)
    cres[i] = ca*cx[i] + cy[i];
}    

template<class T>
void scale_single(T* res, T s, int f_size) {
  for (int i=0;i<f_size;i++)
    res[i] *= s;
}    

template<class T>
void caxpy(T* res, complex<T> ca, T* x, T* y, int f_size) {
  complex<T>* cx = (complex<T>*)x;
  complex<T>* cy = (complex<T>*)y;
  complex<T>* cres = (complex<T>*)res;
  int c_size = f_size / 2;

#pragma omp parallel for
  for (int i=0;i<c_size;i++)
    cres[i] = ca*cx[i] + cy[i];
}    

template<class T>
complex<T> sp_single(T* a, T* b, int f_size) {
  complex<T>* ca = (complex<T>*)a;
  complex<T>* cb = (complex<T>*)b;
  int c_size = f_size / 2;

  int i;
  complex<T> ret = 0.0;
  for (i=0;i<c_size;i++)
    ret += conj(ca[i]) * cb[i];

  return ret;
}

template<class T>
complex<T> sp(T* a, T* b, int f_size) {
  complex<T>* ca = (complex<T>*)a;
  complex<T>* cb = (complex<T>*)b;
  int c_size = f_size / 2;

  complex<T> res = 0.0;
#pragma omp parallel shared(res)
  {
    complex<T> resl = 0.0;
#pragma omp for
    for (int i=0;i<c_size;i++)
      resl += conj(ca[i]) * cb[i];

#pragma omp critical
    {
      res += resl;
    }
  }
  return res;
}

template<class T>
T norm_of_evec(vector< vector<T> >& v, int j) {
  T gg = 0.0;
#pragma omp parallel shared(gg)
  {
    T ggl = 0.0;
#pragma omp for
    for (int nb=0;nb<args.blocks;nb++) {
      T* res = &v[nb][ (int64_t)f_size_block * j ];
      ggl += sp_single(res,res,f_size_block).real();
    }

#pragma omp critical
    {
      gg += ggl;
    }
  }
  return gg;
}

void write_bytes(void* buf, int64_t s, FILE* f, uint32_t& crc) {
  static double data_counter = 0.0;

  // checksum
  crc = crc32_fast(buf,s,crc);

  double t0 = dclock();
  if (fwrite(buf,s,1,f) != 1) {
    fprintf(stderr,"Write failed!\n");
    exit(2);
  }
  double t1 = dclock();

  data_counter += (double)s;
  if (data_counter > 1024.*1024.*256) {
    printf("Writing at %g GB/s\n",(double)s / 1024./1024./1024. / (t1-t0));
    data_counter = 0.0;
  }
}

void write_floats(FILE* f, uint32_t& crc, OPT* in, int64_t n) {
  float* buf = (float*)malloc( sizeof(float) * n );
  if (!buf) {
    fprintf(stderr,"Out of mem\n");
    exit(1);
  }

  // convert to float if needed
#pragma omp parallel for
  for (int64_t i=0;i<n;i++)
    buf[i] = in[i];

  fix_float_endian(buf, n);

  write_bytes(buf,n*sizeof(float),f,crc);

  free(buf);
}

void read_floats(char* & ptr, OPT* out, int64_t n) {
  float* in = (float*)ptr;
  ptr += 4*n;

  for (int64_t i=0;i<n;i++)
    out[i] = in[i];
}

int fp_map(float in, float min, float max, int N) {
  // Idea:
  //
  // min=-6
  // max=6
  //
  // N=1
  // [-6,0] -> 0, [0,6] -> 1;  reconstruct 0 -> -3, 1-> 3
  //
  // N=2
  // [-6,-2] -> 0, [-2,2] -> 1, [2,6] -> 2;  reconstruct 0 -> -4, 1->0, 2->4
  int ret =  (int) ( (float)(N+1) * ( (in - min) / (max - min) ) );
  if (ret == N+1) {
    ret = N;
  }
  return ret;
}

float fp_unmap(int val, float min, float max, int N) {
  return min + (float)(val + 0.5) * (max - min)  / (float)( N + 1 );
}

#define SHRT_UMAX 65535

// can assume that v >=0 and need to guarantee that unmap_fp16_exp(map_fp16_exp(v)) >= v
unsigned short map_fp16_exp(float v) {
  // float has exponents 10^{-44.85} .. 10^{38.53}
#define BASE 1.4142135623730950488
  int exp = (int)ceil(log(v) / log(BASE)) + SHRT_UMAX / 2;
  if (exp < 0 || exp > SHRT_UMAX) {
    fprintf(stderr,"Error in map_fp16_exp(%g,%d)\n",v,exp);
    exit(3);
  }

  return (unsigned short)exp;
}

float unmap_fp16_exp(unsigned short e) {
  float de = (float)((int)e - SHRT_UMAX / 2);
  return pow( BASE, de );
}

void read_floats_fp16(char* & ptr, OPT* out, int64_t n, int nsc) {


  int64_t nsites = n / nsc;
  if (n % nsc) {
    fprintf(stderr,"Invalid size in write_floats_fp16\n");
    exit(4);
  }

  unsigned short* in = (unsigned short*)ptr;
  ptr += 2*(n+nsites);

#define assert(exp)  { if ( !(exp) ) { fprintf(stderr,"Assert " #exp " failed\n"); exit(84); } }
  
  // do for each site
  for (int64_t site = 0;site<nsites;site++) {

    OPT* ev = &out[site*nsc];

    unsigned short* bptr = &in[site*(nsc + 1)];

    unsigned short exp = *bptr++;
    OPT max = unmap_fp16_exp(exp);
    OPT min = -max;

    for (int i=0;i<nsc;i++) {
      ev[i] = fp_unmap( *bptr++, min, max, SHRT_UMAX );
    }

  }

}



int read_meta(char* root, _evc_meta_& args) {

  char buf[1024];
  char val[1024];
  char line[1024];


  // read meta data
  sprintf(buf,"%s.meta",root);
  FILE* f = fopen(buf,"rt");
  if (!f) {
    fprintf(stderr,"Could not open %s\n",buf);
    return 3;
  }

  while (!feof(f)) {
    int i;
    
    if (!fgets(line,sizeof(line),f))
      break;
    
    if (sscanf(line,"%s = %s\n",buf,val) == 2) {
      
      char* r = strchr(buf,'[');
      if (r) {
	*r = '\0';
	i = atoi(r+1);
	
#define PARSE_ARRAY(n) \	
	if (!strcmp(buf,#n)) {			\
	  args.n[i] = atoi(val);		\
	}

	PARSE_ARRAY(s) else
	  PARSE_ARRAY(b) else
	    PARSE_ARRAY(nb) else
	      {
		fprintf(stderr,"Unknown array '%s' in %s.meta\n",buf,root);
		return 4;
	      }

      } else {

#define PARSE_INT(n)				\
	if (!strcmp(buf,#n)) {			\
	  args.n = atoi(val);			\
	}
#define PARSE_HEX(n)				\
	if (!strcmp(buf,#n)) {			\
	  sscanf(val,"%X",&args.n);		\
	}

	PARSE_INT(neig) else
	  PARSE_INT(nkeep) else
	    PARSE_INT(nkeep_single) else
	      PARSE_INT(blocks) else
		PARSE_INT(FP16_COEF_EXP_SHARE_FLOATS) else
		  PARSE_INT(index) else
		    PARSE_HEX(crc32) else
		      {
			fprintf(stderr,"Unknown parameter '%s' in %s.meta\n",buf,root);
			return 4;
		      }
	
	
      }
      
    } else {
      printf("Improper format: %s\n",line); // double nl is OK
    }
    
  }
  
  fclose(f);

  return 1;
}

int main(int argc, char* argv[]) {

#pragma omp parallel
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
    }
  }

  printf("%s\n%d threads\n\n",header,nthreads);

  if (argc < 1+1) {
    fprintf(stderr,"Arguments: fileroot\n");
    return 1;
  }

  char* root = argv[1];
  {
    int i;
    if (!read_meta(root,args))
      return -1;

    printf("Parameters:\n");
    for (i=0;i<5;i++)
      printf("s[%d] = %d\n",i,args.s[i]);
    for (i=0;i<5;i++)
      printf("b[%d] = %d\n",i,args.b[i]);
    printf("nkeep = %d\n",args.nkeep);
    printf("nkeep_single = %d\n",args.nkeep_single);

    vol4d = args.s[0] * args.s[1] * args.s[2] * args.s[3];
    vol5d = vol4d * args.s[4];
    f_size = vol5d / 2 * 24;

    printf("f_size = %d\n",f_size);
    printf("FP16_COEF_EXP_SHARE_FLOATS = %d\n",args.FP16_COEF_EXP_SHARE_FLOATS);
    printf("crc32 = %X\n",args.crc32);
    printf("\n");

    // sanity check
    args.blocks = 1;
    for (i=0;i<5;i++) {
      if (args.s[i] % args.b[i]) {
	fprintf(stderr,"Invalid blocking in dimension %d\n",i);
	return 72;
      }

      args.nb[i] = args.s[i] / args.b[i];
      args.blocks *= args.nb[i];
    }

    f_size_block = f_size / args.blocks;

    printf("number of blocks = %d\n",args.blocks);
    printf("f_size_block = %d\n",f_size_block);

    printf("Internally using sizeof(OPT) = %d\n",sizeof(OPT));

    printf("\n");

    nkeep_fp16 = args.nkeep - args.nkeep_single;
    if (nkeep_fp16 < 0)
      nkeep_fp16 = 0;

    f_size_coef_block = args.neig * 2 * args.nkeep;
  }

  {
    char buf[1024];
    off_t size;

    sprintf(buf,"%s.compressed",root);
    FILE* f = fopen(buf,"r+b");
    if (!f) {
      fprintf(stderr,"Could not open %s\n",buf);
      return 3;
    }

    fseeko(f,0,SEEK_END);

    size = ftello(f);

    fseeko(f,0,SEEK_SET);

    double size_in_gb = (double)size / 1024. / 1024. / 1024.;
    printf("Compressed file is %g GB\n",size_in_gb);

    raw_in = (char*)malloc( size );
    if (!raw_in) {
      fprintf(stderr,"Out of mem\n");
      return 5;
    }

    double t0 = dclock();

    if (fread(raw_in,size,1,f) != 1) {
      fprintf(stderr,"Invalid fread\n");
      return 6;
    }

    double t1 = dclock();

    printf("Read %.4g GB in %.4g seconds at %.4g GB/s\n",
	   size_in_gb, t1-t0,size_in_gb / (t1-t0) );

    uint32_t crc_comp = crc32_fast(raw_in,size,0);

    double t2 = dclock();

    printf("Computed CRC32: %X   (in %.4g seconds)\n",crc_comp,t2-t1);
    printf("Expected CRC32: %X\n",args.crc32);

    if (crc_comp != args.crc32) {
      fprintf(stderr,"Corrupted file!\n");
      return 9;
    }
    
    fclose(f);
  }

  {
    // allocate memory before decompressing
    double uncomp_opt_size = 0.0;
    block_data_ortho.resize(args.blocks);
    for (int i=0;i<args.blocks;i++) {
      block_data_ortho[i].resize(f_size_block * args.nkeep);    
      uncomp_opt_size += (double)f_size_block * args.nkeep;
    }
    block_coef.resize(args.blocks);
    for (int i=0;i<args.blocks;i++) {
      block_coef[i].resize(f_size_coef_block);    
      uncomp_opt_size += (double)f_size_coef_block;
    }
    double t0 = dclock();

    // read
#define FP_16_SIZE(a,b)  (( (a) + (a/b) )*2)
    int _t = (int64_t)f_size_block * (args.nkeep - nkeep_fp16);
    
    //char* ptr = raw_in;
#pragma omp parallel
    {
#pragma omp for  
      for (int nb=0;nb<args.blocks;nb++) {
	char* ptr = raw_in + nb*_t*4;
	read_floats(ptr,  &block_data_ortho[nb][0], _t );
      }
#pragma omp for
      for (int nb=0;nb<args.blocks;nb++) {
	char* ptr = raw_in + args.blocks*_t*4 + FP_16_SIZE( (int64_t)f_size_block * nkeep_fp16 , 24 ) * nb;
	read_floats_fp16(ptr,  &block_data_ortho[nb][ _t ], (int64_t)f_size_block * nkeep_fp16, 24 );
      }
    }
    
    double t1 = dclock();

    char* raw_in_coef = raw_in + args.blocks*_t*4 + FP_16_SIZE( (int64_t)f_size_block * nkeep_fp16 , 24 ) * args.blocks;
    int64_t sz1 = 2*(args.nkeep - nkeep_fp16)*4;
    int64_t sz2 = FP_16_SIZE( 2*nkeep_fp16, args.FP16_COEF_EXP_SHARE_FLOATS);
#pragma omp parallel for
    for (int nb=0;nb<args.blocks;nb++) {
      for (int j=0;j<args.neig;j++) {
	char* ptr = raw_in_coef + (sz1+sz2)*(j * args.blocks + nb);
	read_floats(ptr,  &block_coef[nb][2*args.nkeep*j], 2*(args.nkeep - nkeep_fp16) );
	read_floats_fp16(ptr,  &block_coef[nb][2*args.nkeep*j + 2*(args.nkeep - nkeep_fp16) ], 2*nkeep_fp16 , args.FP16_COEF_EXP_SHARE_FLOATS);
      }
    }

    double t2 = dclock();

    printf("Decompressing single/fp16 to OPT in %g seconds for evec and %g seconds for coefficients; %g GB uncompressed\n",t1-t0,t2-t1,
	   uncomp_opt_size * sizeof(OPT) / 1024./1024./1024.);

  }

  {
    char buf[1024];
    off_t size;
    uint32_t crc32 = 0x0;

    sprintf(buf,"%s.decompressed",root);
    FILE* f = fopen(buf,"w+b");
    if (!f) {
      fprintf(stderr,"Could not open %s\n",buf);
      return 3;
    }

    // now loop through eigenvectors and decompress them
    {
      OPT* dest_all = (OPT*)malloc(f_size * sizeof(OPT) * args.neig);
      if (!dest_all) {
	fprintf(stderr,"Out of mem\n");
	return 33;
      }

      block_data.resize(args.blocks);
      for (int i=0;i<args.blocks;i++)
	block_data[i].resize(f_size_block);    

      double t0 = dclock();
#pragma omp parallel
      {
	for (int j=0;j<args.neig;j++) {

	  OPT* dest = &dest_all[ (int64_t)f_size * j ];

	  double ta,tb;
	  int tid = omp_get_thread_num();

	  if (!tid)
	    ta = dclock();

#if 1

#pragma omp for
	  for (int nb=0;nb<args.blocks;nb++) {
	    
	    OPT* dest_block = &block_data[nb][0];

#if 1
	    {
	      // do reconstruction of this block
	      memset(dest_block,0,sizeof(OPT)*f_size_block);
	      for (int i=0;i<args.nkeep;i++) {
		OPT* ev_i = &block_data_ortho[nb][ (int64_t)f_size_block * i ];
		OPT* coef = &block_coef[nb][ 2*( i + args.nkeep*j ) ];
		caxpy_single(dest_block, *(complex<OPT>*)coef, ev_i, dest_block, f_size_block);
	      }
	    }
#else
	    {
	      complex<OPT>* res = (complex<OPT>*)dest_block;
	      for (int l=0;l<f_size_block/2;l++) {
		complex<OPT> r = 0.0;
		for (int i=0;i<args.nkeep;i++) {
		  complex<OPT>* ev_i = (complex<OPT>*)&block_data_ortho[nb][ (int64_t)f_size_block * i ];
		  complex<OPT>* coef = (complex<OPT>*)&block_coef[nb][ 2*( i + args.nkeep*j ) ];
		  r += coef[i] * ev_i[l];
		}
		res[l] = r;
	      }
	    }
#endif
	  }
#else

	  for (int nb=0;nb<args.blocks;nb++) {
	    
	    OPT* dest_block = &block_data[nb][0];
	    // do reconstruction of this block
#pragma omp for
	    for (int ll=0;ll<f_size_block;ll++)
	      dest_block[ll] = 0;

	    for (int i=0;i<args.nkeep;i++) {
	      OPT* ev_i = &block_data_ortho[nb][ (int64_t)f_size_block * i ];
	      OPT* coef = &block_coef[nb][ 2*( i + args.nkeep*j ) ];
	      caxpy_threaded(dest_block, *(complex<OPT>*)coef, ev_i, dest_block, f_size_block);
	    }
	  }



#endif
	  if (!tid) {
	    tb = dclock();
	    printf("%d - %g seconds\n",j,tb-ta);
	  }
	  
#pragma omp for
	  for (int idx=0;idx<vol4d;idx++) {
	    int pos[5], pos_in_block[5], block_coor[5];
	    index_to_pos(idx,pos,args.s);
	    
	    int parity = (pos[0] + pos[1] + pos[2] + pos[3]) % 2;
	    if (parity == 1) {
	      
	      for (pos[4]=0;pos[4]<args.s[4];pos[4]++) {
		pos_to_blocked_pos(pos,pos_in_block,block_coor);
		
		int bid = pos_to_index(block_coor, args.nb);
		int ii = pos_to_index(pos_in_block, args.b) / 2;
		OPT* dst = &block_data[bid][ii*24];
		
		int co;
		for (co=0;co<12;co++) {
		  OPT* out=&dest[ get_bfm_index(pos,co) ];
		  out[0] = dst[2*co + 0];
		  out[1] = dst[2*co + 1];
		}
	      }
	    }
	  }
	}
      }

      double t1 = dclock();

      printf("Reconstruct eigenvectors in %g seconds\n",t1-t0);

      {
	int i;
	for (i=0;i<args.neig;i++)
	  write_floats(f, crc32, &dest_all[ (int64_t)f_size * i ], f_size);
      }

      double t2 = dclock();

      printf("Wrote data in %g seconds\n",t2-t1);
      
      free(dest_all);
    }

    fclose(f);

    // now write crc32
    sprintf(buf,"%s.decompressed.crc32",root);
    f = fopen(buf,"wt");
    if (!f) {
      fprintf(stderr,"Could not open %s\n",buf);
      return 3;
    }

    fprintf(f,"%X\n",crc32);

    fclose(f);
  }

  free(raw_in);
  return 0;
}
