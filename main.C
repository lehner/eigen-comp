#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex>

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
  int findex;
  int filesperdir;
  int bigendian;
} args;

int vol4d, vol5d;
int f_size, neig;

float* raw_in;

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

complex<float> sp_single(float* a, float* b, int f_size) {
  complex<float>* ca = (complex<float>*)a;
  complex<float>* cb = (complex<float>*)b;
  int c_size = f_size / 2;

  int i;
  complex<float> ret = 0.0;
  for (i=0;i<c_size;i++)
    ret += conj(ca[i]) * cb[i];

  return ret;
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

  if (argc < 1+14) {
    fprintf(stderr,"Arguments: sx sy sz st s5 bx by bz bt b5 nkeep fileindex filesperdir bigendian\n");
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

    printf("Parameters:\n");
    for (i=0;i<5;i++)
      printf("s[%d] = %d\n",i,args.s[i]);
    for (i=0;i<5;i++)
      printf("b[%d] = %d\n",i,args.b[i]);
    printf("nkeep = %d\n",args.nkeep);
    printf("file_index = %10.10d\n",args.findex);
    printf("files_per_dir = %d\n",args.filesperdir);
    printf("big_endian = %d\n",args.bigendian);

    vol4d = args.s[0] * args.s[1] * args.s[2] * args.s[3];
    vol5d = vol4d * args.s[4];
    f_size = vol5d / 2 * 24;

    printf("f_size = %d\n",f_size);

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

    printf("Slot has %d eigenvectors stored\n",neig);

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

  // Have loaded uncompressed eigenvector slot and verified it
  // Next: do the blocking and get coefficients
  



  // Cleanup
  free(raw_in);

  return 0;
}
