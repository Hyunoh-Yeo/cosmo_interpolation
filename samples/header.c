#include<stdio.h>
#include<stddef.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include "pmheader.h"
#include "params.h"
void write_head(FILE *wp, SimParameters wsimpar){
	int isize,ncnt = 0;
	FILE_HEADER(fprintf,wp,,wsimpar);
}
SimParameters read_head(FILE *fp){
	SimParameters rsimpar;
	int isize,ncnt=0;
	char line[MAX_LINE_LENGTH];
	/* Loop until it reaches P_Closing */
	while(strcmp(P_Closing,fgets(line, MAX_LINE_LENGTH,fp)) != 0){
		int ncnt=0;
		/* check whether it is a comment */
		if(line[0] != '#'){
			if(strstr(line,"define") != NULL) FILE_HEADER(sscanf,line,&,rsimpar)
			if(strstr(line,"=") != NULL && strstr(line,"define")!=NULL && ncnt ==0) {
				fprintf(stderr,"Warning: the following parameter line is unknown:\n %s\n\n\n",line);
			}
		}
	}
	/* Now change the stepnum to this step */
	rsimpar.stepnum = rsimpar.stepcount;
	rsimpar.zinit = rsimpar.amax+1.;
	return rsimpar;
}
void write_default_sim_parameter_file(FILE *wp, SimParameters wsimpar){
	int isize,ncnt = 0;
	DEFAULT_PARAMS(fprintf,wp,,wsimpar);
}

void write_sim_parameter_file(FILE *wp, SimParameters wsimpar){
	int isize,ncnt = 0;
	PARAMS(fprintf,wp,,wsimpar);
}
SimParameters  read_sim_parameter_file(FILE *fp){
	SimParameters rsimpar;
	int isize,ncnt=0;
	char line[MAX_LINE_LENGTH];
	while(strcmp(P_Closing,fgets(line, MAX_LINE_LENGTH,fp)) != 0){ /* Loop until it reaches P_Closing */
		int ncnt=0;
		if(line[0] != '#'){ /* check whether it is a comment */
			if(strstr(line,"define") != NULL) PARAMS(sscanf,line,&,rsimpar)
			if(strstr(line,"=") != NULL && strstr(line,"define")!=NULL && ncnt ==0) {
				fprintf(stderr,"Warning: the following parameter line is unknown:\n %s\n\n\n",line);
			}
		}
	}
	rsimpar.stepnum = rsimpar.stepcount; /* Now change the stepnum to this step */
	return rsimpar;
}
void mk_default_param(SimParameters *defsim, char *cosmology){
	double zi;
	defsim->IC = 2;
	/* Simulation Parameters for WMAP 5-year */
	if(strcmp(cosmology,"WMAP3")==0){
		defsim->omep = 0.238;
		defsim->omeplam = 0.762;
		defsim->omepb = 0.042;
		defsim->hubble = 0.732;
		defsim->npow = 0.958;
		defsim->bias = 1.314;
	}
	else {
		defsim->omep = 0.26;
		defsim->omeplam = 0.74;
		defsim->omepb = 0.044;
		defsim->hubble = 0.72;
		defsim->npow = 0.96;
		defsim->bias = 1.26;
	}
	defsim->fNL = 0.;
	defsim->gNL = 0.;
	defsim->boxsize = 1024.;
	defsim->amax = 48.;
	defsim->anow = 1.;
	defsim->nx = defsim->ny = defsim->nz = 1024;
	defsim->nspace = 1;
	defsim->lnx = defsim->lny = defsim->lnz = defsim->nx;
	defsim->mx = defsim->my = defsim->mz = defsim->nx/defsim->nspace;
	defsim->mxmy = defsim->mx*defsim->my;
	defsim->zinit = defsim->amax/defsim->anow-1;
	defsim->theta = 0.3;
	defsim->astep = 0.025;
	defsim->rsmooth = 0.;
	defsim->rth = 8/(defsim->boxsize/defsim->nx);
	defsim->sphere_radius = 4;
	defsim->particle_radius = 4;
	defsim->nstep = 1881;
	defsim->stepcount = defsim->stepnum = 1.;
	defsim->iseed = -56;
	defsim->nskip = 0;
	defsim->powreadflag = 1;
#ifdef XYZDBL
	defsim->xyzshiftflag = 1;
#else
	defsim->xyzshiftflag = 0;
#endif
	sprintf(defsim->rvfilename,"INITIAL");
	sprintf(defsim->rvprefix,"INITIAL");
	sprintf(defsim->powfilename,"camb.z=47.dat");
	zi = defsim->zinit;
	defsim->omei = defsim->omep*pow(1+zi,3)/(defsim->omep*pow(1+zi,3) +
			defsim->omeplam+(1-defsim->omep-defsim->omeplam)*pow(1+zi,2));
	sprintf(defsim->Viewerfile,"\0");
}

/*
#include "mpi.h"
void determine_mpi_misc_param(SimParameters *simpar){
	int nid,myid;
	float zi;
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&nid);
	simpar->nid = nid;
	simpar->myid = myid;
#ifdef XYZDBL
	simpar->xyzshiftflag = 1;
#else
	simpar->xyzshiftflag = 0;
#endif
	zi = simpar->amax-1;
	simpar->omei = simpar->omep*pow(1+zi,3)/(simpar->omep*pow(1+zi,3) +
			simpar->omeplam+(1-simpar->omep-simpar->omeplam)*pow(1+zi,2));
	simpar->mx = simpar->nx/simpar->nspace;
	simpar->my = simpar->ny/simpar->nspace;
	simpar->mz = simpar->nz/simpar->nspace;
	simpar->mxmy = simpar->mx * simpar->my;
	simpar->lnx = simpar->nx;
	simpar->lny = simpar->ny;
	simpar->lnz = simpar->nz;
	simpar->sphere_radius = 4;
	simpar->particle_radius = 4;
	simpar->rth = 8/(simpar->boxsize/simpar->nx);
	simpar->zinit = simpar->amax -1;
}
*/
