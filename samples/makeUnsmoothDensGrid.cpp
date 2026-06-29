#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cstring>

#include<algorithm>
#include<vector>

#include "fof.h"
#include "pmheader.h"
#include "params.h"

#include <H5Cpp.h>

#define	MP	100000

using namespace std;
using namespace H5;

typedef struct Parameter {
	int zSnapshot,numSnapshot;
	int nBox;
} Parameter;

Parameter ReadParameter(int argc,char** argv);
vector<float> GetUnsmoothDensGrid(Parameter params);
SimParameters read_head(FILE* fp);
void UpdateGrid(vector<float>& grid,double x,double y,double z,int nBox,float add);
void WriteDensGrid(vector<float>& grid,Parameter params);

size_t  Fread(void *a,size_t b,size_t c, FILE *fp){
        char *A;
        char t1,t2,t3,t4;
        size_t i,nmem;
        nmem = fread(a,b,c,fp);
        A = (char *)a;
        for(i=0;i<b*nmem;i+=4){
                t1 = A[i];
                t2 = A[i+1];
                t3 = A[i+2];
                t4 = A[i+3];
                A[i] =t4;
                A[i+1] =t3;
                A[i+2] =t2;
                A[i+3] =t1;
        }
        return nmem;
}

#define Fread(a,b,c,d) fread(a,b,c,d)

int main(int argc,char** argv) {

	Parameter params = ReadParameter(argc,argv);

	vector<float> grid = GetUnsmoothDensGrid(params);

	WriteDensGrid(grid,params);
	grid.clear();

	return 1;
}

Parameter ReadParameter(int argc,char** argv) {

	if(argc != 4) {
		puts("Error: ./makeUnsmoothDensGrid.x [boxsize] [id of snapshot] [number of snapshots]");
		exit(-1);
	}

	Parameter params;
	params.nBox = atoi(argv[1]);
	params.zSnapshot = atoi(argv[2]);
	params.numSnapshot = atoi(argv[3]);

	return params;
}

vector<float> GetUnsmoothDensGrid(Parameter params) {

	// 1. Read the simulation parameters
	char strname[200];
	sprintf(strname,"Sync/SyncINITIAL.%05d%05d",params.zSnapshot,0);
	FILE* fp = fopen(strname,"r");
	SimParameters simpar = read_head(fp);
	fclose(fp);
	printf("(boxsize,nx,ny,nz,mx,my,mz) = (%lf,%d,%d,%d,%ld,%ld,%ld)\n",
			simpar.boxsize,simpar.nx,simpar.ny,simpar.nz,simpar.mx,simpar.my,simpar.mz);
	fflush(stdout);

	// 2. Initialize the grid
	int nBox = (int)simpar.boxsize;
	if(nBox != params.nBox) {
		printf("Error: boxsize is different --- %f != %d\n",simpar.boxsize,params.nBox);
		exit(-1);
	}
	int nVol = nBox*nBox*nBox;
	vector<float> grid(nVol,0.);
	float densPerPt =(float)((double)(nVol)/(simpar.nx*simpar.ny*simpar.nz));
	printf("Grid initialization done --- densPerPt = %f\n",densPerPt);
	fflush(stdout);

	READTYPE* arrPtcl = new READTYPE[MP];
	long long ntot = 0L;
	// 3. Iterate over snapshots
	for(int iSnapshot=0;iSnapshot<params.numSnapshot;iSnapshot++) {
		sprintf(strname,"Sync/SyncINITIAL.%05d%05d",params.zSnapshot,iSnapshot);
		fp = fopen(strname,"r");
		simpar = read_head(fp);
		printf("%s ---- done reading header\n",strname);
		printf("%s ---- (boxsize,nx,ny,nz,mx,my,mz) = (%lf,%d,%d,%d,%ld,%ld,%ld)\n",
			strname,simpar.boxsize,simpar.nx,simpar.ny,simpar.nz,simpar.mx,simpar.my,simpar.mz);

		long long np,nall = 0L;
		while((np = Fread(arrPtcl,sizeof(READTYPE),MP,fp)) > 0) {
			for(READTYPE* ptr=arrPtcl;ptr<arrPtcl+np;ptr++) {
				double x = fmod((double)(ptr->x)+(double)(ptr->indx%simpar.mx)+simpar.nx,simpar.nx)*simpar.boxsize/simpar.nx;
				double y = fmod((double)(ptr->y)+(double)((ptr->indx%simpar.mxmy)/simpar.mx)+simpar.ny,simpar.ny)*simpar.boxsize/simpar.ny;
				double z = fmod((double)(ptr->z)+(double)(ptr->indx/simpar.mxmy)+simpar.nz,simpar.nz)*simpar.boxsize/simpar.nz;

				UpdateGrid(grid,x,y,z,nBox,densPerPt);
			}
			nall += np;
			printf("%s ---- done reading %ld particles\n",strname,nall);
			fflush(stdout);
		}
		fclose(fp);
		ntot += nall;
	}
	delete[] arrPtcl;
	printf("Done reading %ld particles\n",ntot);
	return grid;
}

void UpdateGrid(vector<float>& grid,double x,double y,double z,int nBox,float add) {

	// 1. grid that point lays
	int ixc = ((int)floor(x) + nBox) % nBox;
	int iyc = ((int)floor(y) + nBox) % nBox;
	int izc = ((int)floor(z) + nBox) % nBox;
	double dxc = min(x-floor(x),1.-(x-floor(x)));
	double dyc = min(y-floor(y),1.-(y-floor(y)));
	double dzc = min(z-floor(z),1.-(z-floor(z)));
	double frac_xc = 1.-0.5*(0.5-dxc)*(0.5-dxc);
	double frac_yc = 1.-0.5*(0.5-dyc)*(0.5-dyc);
	double frac_zc = 1.-0.5*(0.5-dzc)*(0.5-dzc);

	// 2. nearby grid
	int ixn = (x - floor(x) > 0.5)?((ixc+1) % nBox):((ixc-1+nBox) % nBox);
	int iyn = (y - floor(y) > 0.5)?((iyc+1) % nBox):((iyc-1+nBox) % nBox);
	int izn = (z - floor(z) > 0.5)?((izc+1) % nBox):((izc-1+nBox) % nBox);
	double frac_xn = 0.5*(0.5-dxc)*(0.5-dxc);
	double frac_yn = 0.5*(0.5-dyc)*(0.5-dyc);
	double frac_zn = 0.5*(0.5-dzc)*(0.5-dzc);

	// 3. calculate
	grid[izc+nBox*(iyc+nBox*ixc)] += add*frac_xc*frac_yc*frac_zc;
	grid[izn+nBox*(iyc+nBox*ixc)] += add*frac_xc*frac_yc*frac_zn;
	grid[izc+nBox*(iyn+nBox*ixc)] += add*frac_xc*frac_yn*frac_zc;
	grid[izn+nBox*(iyn+nBox*ixc)] += add*frac_xc*frac_yn*frac_zn;
	grid[izc+nBox*(iyc+nBox*ixn)] += add*frac_xn*frac_yc*frac_zc;
	grid[izn+nBox*(iyc+nBox*ixn)] += add*frac_xn*frac_yc*frac_zn;
	grid[izc+nBox*(iyn+nBox*ixn)] += add*frac_xn*frac_yn*frac_zc;
	grid[izn+nBox*(iyn+nBox*ixn)] += add*frac_xn*frac_yn*frac_zn;
}


void WriteDensGrid(vector<float>& grid,Parameter params) {

	char strname[200];
	sprintf(strname,"grid/unsmooth.%05d.hdf5",params.zSnapshot);
	H5File file(strname,H5F_ACC_TRUNC);

	hsize_t dims[3] = {params.nBox,params.nBox,params.nBox};

  float* newArr = new float[grid.size()];
	copy(grid.begin(),grid.end(),newArr);

	DataSet dataset(file.createDataSet("NormDensity",PredType::NATIVE_FLOAT,DataSpace(3,dims)));
	dataset.write(newArr,PredType::NATIVE_FLOAT);

	delete[] newArr;
	file.close();
}
