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
  "Date: 2017\n";

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
int f_size, neig, f_size_block, f_size_coef_block;

float* raw_in;

#define OPT double

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

    // estimate of compression

    {
      int nkeep_fp16 = args.nkeep - args.nkeep_single;
      if (nkeep_fp16 < 0)
	nkeep_fp16 = 0;

      double size_of_coef_data = (neig * 2 * (double)args.blocks * (args.nkeep_single * 4 + nkeep_fp16 * 2))  / 1024. / 1024. / 1024.;
      double size_of_evec_data = (args.nkeep_single * f_size * 4 + nkeep_fp16 * f_size * 2)  / 1024. / 1024. / 1024.;
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

    printf("Read %.2g GB in %.2g seconds at %.2g GB/s\n",
	   size_in_gb, t1-t0,size_in_gb / (t1-t0) );

    uint32_t crc_comp = crc32_fast(raw_in,(size_t)f_size * neig * sizeof(float),0);

    double t2 = dclock();

    printf("Computed CRC32: %X   (in %.2g seconds)\n",crc_comp,t2-t1);

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

    printf("Created block structure in %.2g seconds\n",t1-t0);

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

#pragma omp parallel for
    for (int nb=0;nb<args.blocks;nb++) {

      for (int iev=0;iev<nevmax;iev++) {

	OPT* orig = &block_data[nb][ (int64_t)f_size_block * iev ];
	OPT* res = &block_data_ortho[nb][ (int64_t)f_size_block * iev ];

	memcpy(res,orig,sizeof(OPT)*f_size_block);
	
	for (int jev=0;jev<iev;jev++) {
	  
	  OPT* ev_j = &block_data_ortho[nb][ (int64_t)f_size_block * jev ];
	  
	  // res = |i> - <j|i> |j>
	  // <j|res>
	  complex<OPT> res_j = sp_single(ev_j,res,f_size_block);
	  caxpy_single(res,- res_j,ev_j,res,f_size_block);
	}

	// normalize
	complex<OPT> nrm = sp_single(res,res,f_size_block);
	scale_single(res, 1.0 / sqrt(nrm.real()),f_size_block);

      }
    }
    
    double t1 = dclock();

    printf("Gram-Schmidt took %.2g seconds\n",t1-t0);

  }


  // Get coefficients and create graphs
  {
    // create block memory
    block_coef.resize(args.blocks);
    for (int i=0;i<args.blocks;i++)
      block_coef[i].resize(f_size_coef_block);    

    double t0 = dclock();

    for (int j=0;j<neig;j++) {

      double norm_j = norm_of_evec(block_data,j);

      for (int i=0;i<args.nkeep;i++) {

	if (i == j && !(i % args.vrb_nkeep_res))
	  printf("nkeep_residuum %d = %g\n",i,norm_of_evec(block_data,j) / norm_j);

#pragma omp parallel for
	for (int nb=0;nb<args.blocks;nb++) {
	  OPT* res = &block_data[nb][ (int64_t)f_size_block * j ];
	  OPT* ev_i = &block_data_ortho[nb][ (int64_t)f_size_block * i ];
	  
	  complex<OPT> res_i = sp_single(ev_i,res,f_size_block);

	  complex<OPT> c = res_i;
	  OPT* cptr = &block_coef[nb][ 2*( i + args.nkeep*j ) ];
	  cptr[0] = c.real();
	  cptr[1] = c.imag();

	  caxpy_single(res,- c,ev_i,res,f_size_block);
        }
	
      }

      if (!(j % args.vrb_evec_res))
	printf("evec_residuum %d = %g\n",j,norm_of_evec(block_data,j) / norm_j);
    }

    double t1 = dclock();
    
    printf("Computing block-coefficients took %.2g seconds\n",t1-t0);

  }

  // TODO: write out block_data_ortho and block_coef, preferrably in FP16 (could make this optional, another factor of 2 in compression)
  // for FP16 we need to look at variation of exponents of coefficients; how many can we group?
  

  // Cleanup
  free(raw_in);

  return 0;
}
