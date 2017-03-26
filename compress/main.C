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

#define MAX_EVEC_PRINT_NORM 8
#define FP16_WIDTH_24 2.0833333333333
#define FP16_COEF_EXP_SHARE_FLOATS 10
#define FP16_WIDTH_COEF (2.0*(1.0 + 1.0 / (double)FP16_COEF_EXP_SHARE_FLOATS))

int nthreads;
uint32_t crc32_fast(const void* data, size_t length, uint32_t previousCrc32);

inline double dclock() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (1.0*tv.tv_usec + 1.0e6*tv.tv_sec) / 1.0e6;
}

const char* header = 
  "QCD eigenvector compressor\n"
  "Authors: Christoph Lehner\n"
  "Date: 2017\n"
  "\n"
  "CXXFLAGS = " CXXFLAGS "\n"
  "\n";

struct {
  int s[5];
  int b[5];
  int nkeep;

  int nkeep_single;

  int findex;
  int filesperdir;
  int bigendian;

  int vrb_nkeep_res;
  int vrb_evec_res;

  // derived
  int nb[5];
  int blocks;
} args;

int vol4d, vol5d;
int f_size, neig, f_size_block, f_size_coef_block, nkeep_fp16;

float* raw_in;

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

  if ((args.bigendian && machine_is_little_endian) ||  // written for readability
      (!args.bigendian && !machine_is_little_endian)) {
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

  write_bytes(buf,n*sizeof(float),f,crc);

  free(buf);
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

void write_floats_fp16(FILE* f, uint32_t& crc, OPT* in, int64_t n, int nsc) {

  int64_t nsites = n / nsc;
  if (n % nsc) {
    fprintf(stderr,"Invalid size in write_floats_fp16\n");
    exit(4);
  }

  unsigned short* buf = (unsigned short*)malloc( sizeof(short) * (n + nsites) );
  if (!buf) {
    fprintf(stderr,"Out of mem\n");
    exit(1);
  }

  // do for each site
#pragma omp parallel for
  for (int64_t site = 0;site<nsites;site++) {

    OPT* ev = &in[site*nsc];

    unsigned short* bptr = &buf[site*(nsc + 1)];

    OPT max = fabs(ev[0]);
    OPT min;

    for (int i=0;i<nsc;i++) {
      if (ev[i]*ev[i] > max*max)
	max = fabs(ev[i]);
    }

    unsigned short exp = map_fp16_exp(max);
    max = unmap_fp16_exp(exp);
    min = -max;

    *bptr++ = exp;

    for (int i=0;i<nsc;i++) {
      int val = fp_map( ev[i], min, max, SHRT_UMAX );
      if (val < 0 || val > SHRT_UMAX) {
	fprintf(stderr,"Assert failed: val = %d (%d), ev[i] = %.15g, max = %.15g, exp = %d\n",val,SHRT_UMAX,ev[i],max,(int)exp);
	exit(48);
      }
      *bptr++ = (unsigned short)val;
    }

  }

  write_bytes(buf,sizeof(short)*(n + nsites),f,crc);

  free(buf);
}

void get_coef(int nb, int i, int j) {
  OPT* res = &block_data[nb][ (int64_t)f_size_block * j ];
  OPT* ev_i = &block_data_ortho[nb][ (int64_t)f_size_block * i ];
  complex<OPT> c = sp_single(ev_i,res,f_size_block);
  
  OPT* cptr = &block_coef[nb][ 2*( i + args.nkeep*j ) ];
  cptr[0] = c.real();
  cptr[1] = c.imag();
  
  caxpy_single(res,- c,ev_i,res,f_size_block);
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

  if (argc < 1+17) {
    fprintf(stderr,"Arguments: sx sy sz st s5 bx by bz bt b5 nkeep fileindex filesperdir bigendian nkeep_single_prec vrb_nkeep_res vrb_evec_res\n");
    return 1;
  }

  {
    int i;
    for (i=0;i<5;i++) {
      args.s[i] = atoi(argv[1+i]);
      args.b[i] = atoi(argv[6+i]);
    }
    args.nkeep = atoi(argv[11]);
    args.findex = atoi(argv[12]);
    args.filesperdir = atoi(argv[13]);
    args.bigendian = atoi(argv[14]);
    args.nkeep_single = atoi(argv[15]);
    args.vrb_nkeep_res = atoi(argv[16]);
    args.vrb_evec_res = atoi(argv[17]);

    printf("Parameters:\n");
    for (i=0;i<5;i++)
      printf("s[%d] = %d\n",i,args.s[i]);
    for (i=0;i<5;i++)
      printf("b[%d] = %d\n",i,args.b[i]);
    printf("nkeep = %d\n",args.nkeep);
    printf("file_index = %10.10d\n",args.findex);
    printf("files_per_dir = %d\n",args.filesperdir);
    printf("big_endian = %d\n",args.bigendian);
    printf("nkeep_single = %d\n",args.nkeep_single);

    vol4d = args.s[0] * args.s[1] * args.s[2] * args.s[3];
    vol5d = vol4d * args.s[4];
    f_size = vol5d / 2 * 24;

    printf("f_size = %d\n",f_size);

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
  }

  {
    // read slot
    char buf[1024];
    sprintf(buf,"%2.2d/%10.10d",args.findex / args.filesperdir,args.findex);
    FILE* f = fopen(buf,"r+b");
    if (!f) {
      fprintf(stderr,"Could not open %s\n",buf);
      return 1;
    }

    fseeko(f,0,SEEK_END);
    off_t size = ftello(f);

    if ( size % (sizeof(float)*f_size) ) {
      fprintf(stderr,"Invalid file size\n");
      return 2;
    }

    neig = ( size / sizeof(float) / f_size );

    f_size_coef_block = neig * 2 * args.nkeep;

    printf("Slot has %d eigenvectors stored\n",neig);

    printf("Size of operating coefficient data in GB: %g\n", (double)f_size_coef_block * (double)args.blocks / 1024./1024./1024. * sizeof(OPT));

    nkeep_fp16 = args.nkeep - args.nkeep_single;
    if (nkeep_fp16 < 0)
      nkeep_fp16 = 0;

    // estimate of compression

    {
      double size_of_coef_data = (neig * FP16_WIDTH_COEF * (double)args.blocks * (args.nkeep_single * 4 + nkeep_fp16 * FP16_WIDTH_COEF))  / 1024. / 1024. / 1024.;
      double size_of_evec_data = ((args.nkeep - nkeep_fp16)* f_size * 4 + nkeep_fp16 * f_size * FP16_WIDTH_24)  / 1024. / 1024. / 1024.;
      double size_orig = (double)size  / 1024. / 1024. / 1024.;
      double size_of_comp = size_of_coef_data+size_of_evec_data;
      printf("--------------------------------------------------------------------------------\n");
      printf("Original size:     %g GB\n",size_orig);
      printf("Compressed size:   %g GB  (%g GB coef, %g GB evec)\n",size_of_comp,size_of_coef_data,size_of_evec_data);
      printf("Compressed to %.4g%% of original\n",size_of_comp / size_orig * 100.);
      printf("--------------------------------------------------------------------------------\n");
    }
    //

    fseeko(f,0,SEEK_SET);

    raw_in = (float*)malloc( (size_t)f_size * (neig * sizeof(float)) );
    
    if (!raw_in) {
      fprintf(stderr,"Out of mem\n");
      return 5;
    }

    double t0 = dclock();

    if (fread(raw_in,f_size,neig*sizeof(float),f) != neig*sizeof(float)) {
      fprintf(stderr,"Invalid fread\n");
      return 6;
    }

    double t1 = dclock();

    double size_in_gb = f_size * (double)neig * sizeof(float) / 1024. / 1024. / 1024.;

    printf("Read %.4g GB in %.4g seconds at %.4g GB/s\n",
	   size_in_gb, t1-t0,size_in_gb / (t1-t0) );

    uint32_t crc_comp = crc32_fast(raw_in,(size_t)f_size * neig * sizeof(float),0);

    double t2 = dclock();

    printf("Computed CRC32: %X   (in %.4g seconds)\n",crc_comp,t2-t1);

    fclose(f);

    // and checksums
    f = fopen("checksums.txt","rt");
    if (!f) {
      fprintf(stderr,"Could not read checksums\n");
      return 2;
    }

    for (int i=0;i<args.findex+2;i++)
      fgets(buf,sizeof(buf),f);

    uint32_t crc_expect;

    fscanf(f,"%X",&crc_expect);

    fclose(f);

    printf("Expected CRC32: %X\n",crc_expect);

    if (crc_comp != crc_expect) {
      fprintf(stderr,"Corrupted file!\n");
      return 9;
    }

    printf("Fixing endian-ness\n");

    // fix endian if needed
#pragma omp parallel for
    for (int j=0;j<neig;j++) {
      float* segm = &raw_in[ (int64_t)f_size * j ];
      fix_float_endian(segm, f_size);

      if (j<MAX_EVEC_PRINT_NORM)
	printf("Norm %d: %g\n",j, sp_single(segm,segm,f_size).real());
    }


  }

  //
  // Status: 
  //  loaded uncompressed eigenvector slot and verified it
  //

  {

    // create block memory
    block_data.resize(args.blocks);
    for (int i=0;i<args.blocks;i++)
      block_data[i].resize(f_size_block * neig);    

    double t0 = dclock();
    
    //
#pragma omp parallel 
    {
      for (int nev=0;nev<neig;nev++) {
	float* raw_in_ev = &raw_in[ (int64_t)f_size * nev ];
	
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

	      OPT* dst = &block_data[bid][ ii*24 + (int64_t)f_size_block * nev ];
	      
	      int co;
	      for (co=0;co<12;co++) {
		float* in=&raw_in_ev[ get_bfm_index(pos,co) ];
		dst[2*co + 0] = in[0]; // may convert precision depending on OPT
		dst[2*co + 1] = in[1];
	      }
	    }
	  }
	}
      }
    }

    double t1 = dclock();

    printf("Created block structure in %.4g seconds\n",t1-t0);

    // simple test
    {
      int test_ev = neig - 1;
      float* raw_in_ev = &raw_in[ (int64_t)f_size * test_ev ];
      float nrm = sp(raw_in_ev,raw_in_ev,f_size).real();

      int i;
      double nrm_blocks = 0.0;
      for (i=0;i<args.blocks;i++) {
	OPT* in_ev = &block_data[i][ (int64_t)f_size_block * test_ev ];
	nrm_blocks += sp(in_ev,in_ev,f_size_block).real();
      }

      printf("Difference of checksums after blocking: %g - %g = %g\n",nrm,nrm_blocks,nrm-nrm_blocks);

      if (fabs(nrm - nrm_blocks) > 1e-5) {
	fprintf(stderr,"Unexpected error in creating blocks\n");
	return 91;
      }
    }
  }

  // Now do Gram-Schmidt
  {
    // create block memory
    block_data_ortho.resize(args.blocks);
    for (int i=0;i<args.blocks;i++)
      block_data_ortho[i].resize(f_size_block * args.nkeep);    
    
    double t0 = dclock();

    int nevmax = args.nkeep;

    double flops = 0.0;
    double bytes = 0.0;
#pragma omp parallel shared(flops,bytes)
    {
      double flopsl = 0.0;
      double bytesl = 0.0;
#define COUNT_FLOPS_BYTES(f,b) flopsl += (f) + 1; bytesl += (b) + 2;
      // #define COUNT_FLOPS_BYTES(f,b)

#pragma omp for
      for (int nb=0;nb<args.blocks;nb++) {
	
	for (int iev=0;iev<nevmax;iev++) {
	  
	  OPT* orig = &block_data[nb][ (int64_t)f_size_block * iev ];
	  OPT* res = &block_data_ortho[nb][ (int64_t)f_size_block * iev ];
	  
	  memcpy(res,orig,sizeof(OPT)*f_size_block); 	  COUNT_FLOPS_BYTES(f_size_block,2*f_size_block*sizeof(OPT));

	  for (int jev=0;jev<iev;jev++) {
	    
	    OPT* ev_j = &block_data_ortho[nb][ (int64_t)f_size_block * jev ];
	    
	    // res = |i> - <j|i> |j>
	    // <j|res>
	    complex<OPT> res_j = sp_single(ev_j,res,f_size_block);  
	    COUNT_FLOPS_BYTES(8 / 2 * f_size_block, 2*f_size_block*sizeof(OPT)); // 6 per complex multiply, 2 per complex add -> 8 / 2 = 4

	    caxpy_single(res,- res_j,ev_j,res,f_size_block); 
	    COUNT_FLOPS_BYTES(8 / 2 * f_size_block, 3*f_size_block*sizeof(OPT));
	  }
	  
	  // normalize
	  complex<OPT> nrm = sp_single(res,res,f_size_block); 
	  COUNT_FLOPS_BYTES(8 / 2 * f_size_block,2*f_size_block*sizeof(OPT));

	  scale_single(res, (OPT)(1.0 / sqrt(nrm.real())),f_size_block); 
	  COUNT_FLOPS_BYTES(f_size_block,2*f_size_block*sizeof(OPT));
	  
	}
      }

#pragma omp critical
      {
	flops += flopsl + 1;
	bytes += bytesl + 2;
      }
    }
    
    double t1 = dclock();

    printf("Gram-Schmidt took %.4g seconds (%g Gflops/s, %g GB/s)\n",t1-t0,flops / (t1-t0) / 1000./1000./1000.,
	   bytes / (t1-t0) / 1024./1024./1024.);

  }


  // Get coefficients and create graphs
  {
    // create block memory
    block_coef.resize(args.blocks);
    for (int i=0;i<args.blocks;i++)
      block_coef[i].resize(f_size_coef_block);    

    double t0 = dclock();

    if (!args.vrb_nkeep_res && !args.vrb_evec_res) {
      printf("Do not display convergence, use fast codepath for obtaining coefficients\n");

#pragma omp parallel for
      for (int nb=0;nb<args.blocks;nb++) {
	for (int j=0;j<neig;j++) {
	  for (int i=0;i<args.nkeep;i++) {
	    get_coef(nb,i,j);
	  }
	}
      }

    } else {

      printf("Slow codepath to display convergence\n");
      
      for (int j=0;j<neig;j++) {
	
	double norm_j = -1.0;
	
	// only compute norm if needed for verbosity
	if ((j < args.nkeep && args.vrb_nkeep_res < args.nkeep) ||
	    !(j % args.vrb_evec_res))
	  norm_j = norm_of_evec(block_data,j);
	
	for (int i=0;i<args.nkeep;i++) {
	  
	  if (i == j && !(i % args.vrb_nkeep_res))
	    printf("nkeep_residuum %d = %g\n",i,norm_of_evec(block_data,j) / norm_j);
	  
#pragma omp parallel for
	  for (int nb=0;nb<args.blocks;nb++) {
	    get_coef(nb,i,j);
	  }
	  
	}
	
	if (!(j % args.vrb_evec_res))
	  printf("evec_residuum %d = %g\n",j,norm_of_evec(block_data,j) / norm_j);
      }
    }


    double t1 = dclock();
    
    printf("Computing block-coefficients took %.4g seconds\n",t1-t0);

  }

  // write result
  {
    uint32_t crc = 0x0;
    off_t begin_fp16_evec;
    off_t begin_coef;
    char buf[1024];
    
    // write data
    {
      sprintf(buf,"%2.2d/%10.10d.compressed",args.findex / args.filesperdir,args.findex);
      FILE* f = fopen(buf,"w+b");
      if (!f) {
	fprintf(stderr,"Could not open %s for writing!\n",buf);
	return 1;
      }
      
      int nb;
      
      int _t = (int64_t)f_size_block * (args.nkeep - nkeep_fp16);
      for (nb=0;nb<args.blocks;nb++)
	write_floats(f,crc,  &block_data_ortho[nb][0], _t );
      
      begin_fp16_evec = ftello(f);
      
      for (nb=0;nb<args.blocks;nb++)
	write_floats_fp16(f,crc,  &block_data_ortho[nb][ _t ], (int64_t)f_size_block * nkeep_fp16, 24 );
      
      begin_coef = ftello(f);
      
      // write coefficients of args.nkeep_single as floats, higher coefficients as fp16
      
#if 0
      for (int nb=0;nb<2;nb++) {
	
	for (int j = 0; j < 2000; j++) {
	  
	  printf("Coefficients of block %d, eigenvector %d\n",nb,j);
	  
	  for (int i = 105; i < 106; i++) {
	    
	    OPT* cptr = &block_coef[nb][ 2*( i + args.nkeep*j ) ];
	    
	    printf("c[%d] = %g , %g\n",i,cptr[0],cptr[1]);
	  }
	}
      }
#endif
      
      int j;
      for (j=0;j<neig;j++)
	for (nb=0;nb<args.blocks;nb++) {
	  write_floats(f,crc,  &block_coef[nb][2*args.nkeep*j], 2*(args.nkeep - nkeep_fp16) );
	  write_floats_fp16(f,crc,  &block_coef[nb][2*args.nkeep*j + 2*(args.nkeep - nkeep_fp16) ], 2*nkeep_fp16 , FP16_COEF_EXP_SHARE_FLOATS);
	}
      
      fclose(f);
    }

    // write meta data
    {
      sprintf(buf,"%2.2d/%10.10d.meta",args.findex / args.filesperdir,args.findex);
      FILE* f = fopen(buf,"wt");
      if (!f) {
	fprintf(stderr,"Could not open %s for writing!\n",buf);
	return 1;
      }

      fprintf(f,"crc32 = %X\n",crc);
      int i;
      for (i=0;i<5;i++)
	fprintf(f,"s[%d] = %d\n",i,args.s[i]);
      for (i=0;i<5;i++)
	fprintf(f,"b[%d] = %d\n",i,args.b[i]);
      for (i=0;i<5;i++)
	fprintf(f,"nb[%d] = %d\n",i,args.nb[i]);
      fprintf(f,"neig = %d\n",neig);
      fprintf(f,"nkeep = %d\n",args.nkeep);
      fprintf(f,"nkeep_single = %d\n",args.nkeep_single);
      fprintf(f,"blocks = %d\n",args.blocks);
      fprintf(f,"FP16_COEF_EXP_SHARE_FLOATS = %d\n",FP16_COEF_EXP_SHARE_FLOATS);
      fprintf(f,"index = %d\n",args.findex);

      fclose(f);
    }
  }

  // Cleanup
  free(raw_in);

  return 0;
}
