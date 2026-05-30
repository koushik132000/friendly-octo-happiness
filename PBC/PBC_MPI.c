//Program generates equilibrium configurations for 36 chains in 7x7x20 lattice 
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "ran2.h"
#include <mpi.h>
/*******************************************************************************
Folded chains are equilibrated by performing MC moves with an excluded volume constraint and bead-bead interactions. Moves are accepted or rejected based on metropolis algorithm. Generates 1000 configuration files for a system. 
*******************************************************************************/

//Creation Date:27Nov2024
//Last Modified:16May2026

/*
Modifications:
01-12-2024: Introduced subroutine to determine state of a lattice site
06-12-2024: Energy change calculations subroutine introduced
07-12-2024: Direction evaluation subroutine introduced
07-12-2024: Remove array formulations for terminal bead motion
07-12-2024: Introduce overlap energy calculations
08-12-2024: Introduced check for number of beads occupying a site
09-12-2024: Corrected neighborlist calculation in deltaE subroutine
10-01-2025: Writing file after move increment is done
11-01-2025: Write the initial configuration before MC moves are executed
21-02-2025: Updated comments for the program
06-02-2025: Updated for variable number of simulation steps and dumping data every mcsteps
21-10-2025: Corrected energy calculation in deltaE
05-05-2026: Changed the code to incorporate folding without bead overlap
05-05-2026: Detailed comments included
06-05-2026: Modified algorithm for moving terminal beads
07-05-2026: Modified to ensure equal probability of picking all beads
07-05-2026: Separates equilibration and production simulations
08-05-2026: Corrected algorithm for determining available direction for terminal beads
08-05-2026: Corrected trial move for first bead in the available direction
12-05-2026: Implementing PBC in davail
13-05-2026: Implementing PBC with nneigh
15-05-2026: Corrected sflg to pointer variable
16-05-2026: Implemented MPI
*/

//Definitions
#define nc 36//number of chains
#define bc 20//number of beads per chain
#define lx 7//box size in x-direction
#define ly 7//box size in y-direction
#define lz 20//box size in z-direction
#define lxly 49//number of sites in xy layer
#define ns 980//number of sites 

/*******************************************************************************
There are 3 functions other than the main function
davail: Determines availability of a direction for terminal bead moves. Returns integer flag for the specified direction. 
nneigh: Determines number of occupied sites in the neighborhood of a specific lattice site. It also modifies the site flag associated with a bead. Returns integer value for number of occupied sites.
deltaE: Energy change associated with a move of any bead. Returns value of energy difference in double precision.
metrop: Metropolis algorithm returns integer flag for accepting/rejecting moves
*******************************************************************************/

/*******************************************************************************
There are 7 subroutines
sstate: Determine lattice site status and maps it to chain details
fmoves: Moves the first bead of the chain
nmoves: Moves the last bead of the chain
kmoves: Moves kth internal bead of the chain
fmeval: Evaluates energy associated with move of first bead
lmeval: Evaluates energy associated with move of last bead
accmov: Accepts moves and reconfigures the chain
*******************************************************************************/

//Structure for providing x,y,z position of bc beads in a chain 
struct pos{
	int x[bc];
	int y[bc];
	int z[bc]; 
};

//Structure for providing components of direction vectors 
struct vec{
	int ex;
	int ey;
	int ez;
};

//Structure for providing chain and bead associations with ns lattice sites  
struct site{
	int sx[ns];
	int sy[ns];
	int sz[ns];
};

//Array of structure giving bc bead positions of nc different chains
struct pos beads[nc];

//Structure provides occupation details of ns lattice sites
struct site lcs;
//Unoccupied sites have chain=0, bead=0 and number of beads occupying=0

//Defines unit vectors for the directions in the cubic lattice
struct vec dir[6]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

//Temporary position vector components for an attempted move
struct vec fpos;//First bead components
struct vec npos;//Last bead components
struct vec kpos;//Chosen internal bead components

//Inverses needed for PBC calculations
double invx=1.0/lx;
double invy=1.0/ly;
double invz=1.0/lz;

//Specific indices for chain end beads
int nb=bc-1;//index of the last bead
int pb=bc-2;//index of the penultimate bead
int cb=bc-3;//index of bead connected to penultimate bead

/*******************************************************************************
A bead occupies a site with a specific site index j=0,1,...,719. A bead that occupies a site with index pcalc can move to a new site with site index scalc.
*******************************************************************************/
int scalc;//Calculated site index
int pcalc;//Previous site index

//Site index of site occupied by first and last bead of a chain
int fcalc,lcalc;

//Energy change when the first, last and internal beads of a chain are moved
double dEf,dEl,dE;

/*******************************************************************************
For folding a chain we do not consider bead-bead and bead-wall interaction. A large penalty is imposed when a bead attempts to move into an already occupied lattice site.
*******************************************************************************/
double Ea=0.0;//Bead-Wall interaction energy
double Eb=-0.457;//Bead-Bead interaction energy
double Ex=200.0;//Bead-Bead overlap energy

long seed;//seed for the ran2 generator
	
//Determines lattice site details
void sstate()
{
	int i,j,k;
	int xval,yval,zval;

	//Initialize lattice site status and details
	for (k=0;k<ns;k++){
		lcs.sx[k]=0;
		lcs.sy[k]=0;
		lcs.sz[k]=0;
	}

	//Update chain, bead and occupation status details of a lattice site
	//Loop over all the chains
	for (i=0;i<nc;i++)
	{
		//Loop over all the beads of a chain
		for (j=0;j<bc;j++)
		{
			xval=beads[i].x[j];
			yval=beads[i].y[j];
			zval=beads[i].z[j];			
			k=xval+lx*yval+lx*ly*zval;//Site index calculation
			lcs.sx[k]=i;//Chain occupying the lattice
			lcs.sy[k]=j;//Bead occupying the lattice
			lcs.sz[k]+=1;//Number of beads occupying a lattice
		}
	}
	
}

/*******************************************************************************
In function davail a protocol for determining the availability of the direction for moving a terminal bead is provided. It uses the following 4 step protocol
1) Determine the segment vectors for two terminal segments.
2) Determine the relative orientation of the terminal segments with reference to the chosen direction.
3) Determine the relative orientation of the two terminal segments.
4) Determine the availability of the move based on available directions.
Returns avail=0 if direction is not available 
Returns avail=1 if direction is available
*******************************************************************************/

//Determine if a move direction is available for terminal beads
int davail(int cnum,int r,int tflag)
{
	int avail;

	//Terminal segment vectors
	struct vec vec1;
	struct vec vec2;
	
	//Relative orientation of the terminal segments
	int dota1,dota2,dota3;

	//Determine segment vectors for two terminal segments
	if (tflag==0)//First bead
	{
		//Determing the current orientation of the first segment
		vec1.ex=beads[cnum].x[0]-beads[cnum].x[1];
		vec1.ey=beads[cnum].y[0]-beads[cnum].y[1];
		vec1.ez=beads[cnum].z[0]-beads[cnum].z[1];
		
		//Determing the current orientation of the second segment
		vec2.ex=beads[cnum].x[1]-beads[cnum].x[2];
		vec2.ey=beads[cnum].y[1]-beads[cnum].y[2];
		vec2.ez=beads[cnum].z[1]-beads[cnum].z[2];
		
	}
	else//Last bead
	{
		//Determing the current orientation of the last segment
		vec1.ex=beads[cnum].x[nb]-beads[cnum].x[pb];
		vec1.ey=beads[cnum].y[nb]-beads[cnum].y[pb];
		vec1.ez=beads[cnum].z[nb]-beads[cnum].z[pb];	
		
		//Determing the current orientation of the penultimate segment
		vec2.ex=beads[cnum].x[pb]-beads[cnum].x[cb];
		vec2.ey=beads[cnum].y[pb]-beads[cnum].y[cb];
		vec2.ez=beads[cnum].z[pb]-beads[cnum].z[cb];	
	}

	//PBC displacement
	vec1.ex-=lx*round(vec1.ex*invx);
	vec2.ex-=lx*round(vec2.ex*invx);
	vec1.ey-=ly*round(vec1.ey*invy);
	vec2.ey-=ly*round(vec2.ey*invy);
	vec1.ez-=lz*round(vec1.ez*invz);
	vec2.ez-=lz*round(vec2.ez*invz);
	
	if(vec1.ex>1||vec1.ex<-1||vec2.ex>1||vec2.ex<-1)
        printf("Error\n");
	if(vec1.ey>1||vec1.ey<-1||vec2.ey>1||vec2.ey<-1)
		printf("Error\n");
	if(vec1.ez>1||vec1.ez<-1||vec2.ez>1||vec2.ez<-1)
		printf("Error\n");
	
	//Determine relative orientations of the terminal segments
	dota1=vec1.ex*dir[r].ex+vec1.ey*dir[r].ey+vec1.ez*dir[r].ez;
	dota2=vec2.ex*dir[r].ex+vec2.ey*dir[r].ey+vec2.ez*dir[r].ez;
	dota3=vec1.ex*vec2.ex+vec1.ey*vec2.ey+vec1.ez*vec2.ez;

	avail=0;//Initialize availability
	if (dota3==1 && dota1==0 && dota2==0)
		avail=1;
	else if (dota3==0 && dota1<=0 && dota2>=0)
		avail=1;

	return avail;
}

/*******************************************************************************
In subroutines fmoves and lmoves the beads are constrained to move within the cubic box boundary. Position values can not be negative and nor can they exceed lx-1, ly-1 and lz-1.
*******************************************************************************/
//Trial moves for the first bead of the chain
void fmoves(int cnum,int r)
{
	int reset;
		
	//New trial position for the first bead
	fpos.ex=beads[cnum].x[1]+dir[r].ex;
	fpos.ey=beads[cnum].y[1]+dir[r].ey;
	fpos.ez=beads[cnum].z[1]+dir[r].ez;	
	
	//Applying PBC boundary conditions
	//x direction
	if (fpos.ex<0)
		fpos.ex=lx-1;
	if (fpos.ex>lx-1)
		fpos.ex=0;
	//y direction
	if (fpos.ey<0)
		fpos.ey=ly-1;
	if (fpos.ey>ly-1)
		fpos.ey=0;
	//z direction
	if (fpos.ez<0)
		fpos.ez=lz-1;
	if (fpos.ez>lz-1)
		fpos.ez=0;

	//Prevent beads leaving the simulation box
	if (fpos.ex<0||fpos.ey<0||fpos.ez<0||fpos.ex>lx-1||fpos.ey>ly-1||fpos.ez>lz-1)
		reset=1;
	else
		reset=0;
	
	if (reset==1)
	{
		fpos.ex=beads[cnum].x[0];
		fpos.ey=beads[cnum].y[0];
		fpos.ez=beads[cnum].z[0];	
	}
}

//Trial moves for the last bead of the chain
void nmoves(int cnum,int r)
{
	int move,reset;
	struct vec nvec;

	//New trial position for the last bead
	npos.ex=beads[cnum].x[pb]+dir[r].ex;
	npos.ey=beads[cnum].y[pb]+dir[r].ey;
	npos.ez=beads[cnum].z[pb]+dir[r].ez;	

	//Applying PBC boundary conditions
	//x direction
	if (npos.ex<0)
		npos.ex=lx-1;
	if (npos.ex>lx-1)
		npos.ex=0;
	//y direction
	if (npos.ey<0)
		npos.ey=ly-1;
	if (npos.ey>ly-1)
		npos.ey=0;
	//z direction
	if (npos.ez<0)
		npos.ez=lz-1;
	if (npos.ez>lz-1)
		npos.ez=0;

	//Prevent beads leaving the simulation box
	if (npos.ex<0||npos.ey<0||npos.ez<0||npos.ex>lx-1||npos.ey>ly-1||npos.ez>lz-1)
		reset=1;
	else
		reset=0;

	if (reset==1){
		npos.ex=beads[cnum].x[nb];
		npos.ey=beads[cnum].y[nb];
		npos.ez=beads[cnum].z[nb];
	}

}

/*******************************************************************************
Kink flip protocol is used for deciding trial moves of the internal beads. The following 4 step protocol is used:
1) Determine the direction of bond segments that connect bead k to bead k+1 denoted by vec1 and k-1 denoted by vec2. 
2) Check if the direction of the segments vec1 and vec2 is diferent. Kink flip can be performed only when this is true.
3) Kink flip is unique and is performed by setting vec1=vec2 and vec2=vec1. This is achieved by changing the position of bead k such that vec2 is replaced by vec1.
4) The beads are constrained to move within the cubic box boundary. Position values can not be negative and nor can they exceed lx-1, ly-1 and lz-1.
*******************************************************************************/

//Trial moves for the kth non-terminal bead of a chain
void kmoves(int cnum, int k)
{
	int move,reset;

	//Bond segment vectors
	struct vec vec1;
	struct vec vec2;

	//Relative orientation of segment vectors
	int dota;

	//Determing the current orientation of the vec1 segment
	vec1.ex=beads[cnum].x[k+1]-beads[cnum].x[k];
	vec1.ey=beads[cnum].y[k+1]-beads[cnum].y[k];
	vec1.ez=beads[cnum].z[k+1]-beads[cnum].z[k];
	
	//Determing the current orientation of the vec2 segement
	vec2.ex=beads[cnum].x[k]-beads[cnum].x[k-1];
	vec2.ey=beads[cnum].y[k]-beads[cnum].y[k-1];
	vec2.ez=beads[cnum].z[k]-beads[cnum].z[k-1];
		
	//PBC displacement
	vec1.ex-=lx*round(vec1.ex*invx);
	vec2.ex-=lx*round(vec2.ex*invx);
	vec1.ey-=ly*round(vec1.ey*invy);
	vec2.ey-=ly*round(vec2.ey*invy);
	vec1.ez-=lz*round(vec1.ez*invz);
	vec2.ez-=lz*round(vec2.ez*invz);
	
	if(vec1.ex>1||vec1.ex<-1||vec2.ex>1||vec2.ex<-1)
		printf("Error\n");
	if(vec1.ey>1||vec1.ey<-1||vec2.ey>1||vec2.ey<-1)
		printf("Error\n");
	if(vec1.ez>1||vec1.ez<-1||vec2.ez>1||vec2.ez<-1)
		printf("Error\n");

	//Move is allowed when there is a kink between k and k-1 segments
	dota=vec1.ex*vec2.ex+vec1.ey*vec2.ey+vec1.ez*vec2.ez;
	if (dota!=0)  
		move=0;
	else
		move=1;
	
	//Moving the chosen bead in the only allowed direction
	if (move==1){
		kpos.ex=beads[cnum].x[k-1]+vec1.ex;
		kpos.ey=beads[cnum].y[k-1]+vec1.ey;
		kpos.ez=beads[cnum].z[k-1]+vec1.ez;	
	}
	else{
		kpos.ex=beads[cnum].x[k];
		kpos.ey=beads[cnum].y[k];
		kpos.ez=beads[cnum].z[k];	
	}

	//Applying PBC boundary conditions
	//x direction
	if (kpos.ex<0)
		kpos.ex=lx-1;
	if (kpos.ex>lx-1)
		kpos.ex=0;
	//y direction
	if (kpos.ey<0)
		kpos.ey=ly-1;
	if (kpos.ey>ly-1)
		kpos.ey=0;
	//z direction
	if (kpos.ez<0)
		kpos.ez=lz-1;
	if (kpos.ez>lz-1)
		kpos.ez=0;

	//Prevent beads leaving the simulation box
	if (kpos.ex<0||kpos.ey<0||kpos.ez<0||kpos.ex>lx-1||kpos.ey>ly-1||kpos.ez>lz-1)
		reset=1;
	else
		reset=0;

	if (reset==1){
		kpos.ex=beads[cnum].x[k];
		kpos.ey=beads[cnum].y[k];
		kpos.ez=beads[cnum].z[k];
	}

}

/*******************************************************************************
A site always has a maximum of 6 neighbors
*******************************************************************************/
//Number of neighbors calculation
int nneigh(int sindex, int *sflg)
{
	int nns;//Number of neighboring sites
	int ons;//Number of occupied sites
	int xl,yl,zl;//Layers associated with a site
	int nindex;//Neighbor site index
	int r;

	//Layers associted with a site
	xl=sindex%lx;//x layer
	yl=((sindex-xl)/lx)%ly;//y layer
	zl=((sindex-xl-yl*lx)/lxly)%lz;//z layer

	/***********************************************************************
	Layer index is used to distinguish surface, edge and corner sites
	***********************************************************************/
	//*sflg=0 is internal, 1 is wall, 2 is edge and 3 is corner
    *sflg=0;//Initialize all sites as internals
	/***********************************************************************
	Calculate using layer values
	1) Number of neighbor sites to sindex - nns
	2) Number of occupied nighbor sites to sindex - ons
	***********************************************************************/
	ons=0;//Initialized number of occupied sites
	nns=0;//Initialize number of neighbors sites
	//Loop over all directions
	for (r=0;r<6;r++)
	{
		if(r==0 && xl!=0)
		{
			nindex=sindex-1;
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		else if(r==0 && xl==0)
		{
			nindex=sindex+lx-1;
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		if(r==1 && xl!=lx-1)
		{
			nindex=sindex+1;
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		else if(r==1 && xl==lx-1)
		{
			nindex=sindex-(lx-1);
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		if(r==2 && yl!=0)
		{
			nindex=sindex-lx;
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		else if(r==2 && yl==0)
		{
			nindex=sindex+lx*(ly-1);
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		if(r==3 && yl!=ly-1)
		{
			nindex=sindex+lx;
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		else if(r==3 && yl==ly-1)
		{
			nindex=sindex-lx*(ly-1);
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		if(r==4 && zl!=0)
		{
			nindex=sindex-lxly;
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		else if(r==4 && zl==0)
		{
			nindex=sindex+lxly*(lz-1);
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		if(r==5 && zl!=lz-1)
		{
			nindex=sindex+lxly;
			nns+=1;
			ons+=lcs.sz[nindex];
		}
		else if(r==5 && zl==lz-1)
		{
			nindex=sindex-lxly*(lz-1);
			nns+=1;
			ons+=lcs.sz[nindex];
		}	
	}

	//Verify Results
	if (*sflg==0 && nns!=6)
		printf("Error in internal neighbor calculation: %d\n",nns);
	if(ons>nns)
		printf("Error in occupied site calculation\n");

	return ons;
}

//Energy change calculation for a trial move
double deltaE(int olds,int news)
{
	/***********************************************************************
	olds is the index of the site occupied currently
	news is the index of the site identified for trial move
	***********************************************************************/
	
	double Eold=0,Enew=0;//Energy values in the old and new site
	double Ediff;//Energy difference due to the trial move
    int osflg,nsflg;//Old and new site flag for determining surface proximity
	int osv,nsv;//Number of neighbors for the old and new site
	
	//Initialize energy difference
	Ediff=0.0;
	
	//Energy zero for single occupancy and increases for higher occupancy
	Eold=(lcs.sz[olds]-1)*Ex;
	if (Eold!=0.0)
		printf("Old site is occupied by more than one bead\n");

	//Energy penalty for moving to an occupied site
	if(lcs.sz[news]>=1)
		Enew = lcs.sz[news]*Ex;
	
	//Bead-Bead interaction energy contributions
	osv=nneigh(olds,&osflg);
	Eold+=osv*Eb;//Bead-Bead
	nsv=nneigh(news,&nsflg);
	Enew+=nsv*Eb;//Bead-Bead

	//Change in energy due to the move
	Ediff=Enew-Eold;

	return Ediff;
}

//Evaluates the terminal moves for first bead in a specific direction	
void fmeval(int cnum,int dindex,int fcalc)
{
	fmoves(cnum,dindex);
	scalc=fpos.ex+lx*fpos.ey+lx*ly*fpos.ez;
	dEf=deltaE(fcalc,scalc);
}	

//Evaluates the terminal move for the last bead in a specific direction
void lmeval(int cnum,int dindex,int lcalc)
{
	nmoves(cnum,dindex);
	scalc=npos.ex+lx*npos.ey+lx*ly*npos.ez;
	dEl=deltaE(lcalc,scalc);
}

//Metropolis algorithm implementation
int metrop(double delE)
{
	int acc;
	double rij;
	double pij;

	if (delE<=0){
		pij=1.0;
		acc=1;
	}
	else{
		pij=exp(-delE);
		rij=ran2(&seed);
		if (pij<rij)
			acc=0;
		else
			acc=1;
	}
	return acc;
}

//Accepts move and reconfigures the chain
void accmov(int cnum,int bnum,int pcalc,int scalc,struct vec mpos)
{
	lcs.sx[pcalc]=0;
	lcs.sy[pcalc]=0;
	lcs.sz[pcalc]-=1;
	beads[cnum].x[bnum]=mpos.ex;
	beads[cnum].y[bnum]=mpos.ey;
	beads[cnum].z[bnum]=mpos.ez;
	lcs.sx[scalc]=cnum;
	lcs.sy[scalc]=bnum;
	lcs.sz[scalc]+=1;
}

//Main simulation
int main(int argc, char **argv)
{
    //Initialize MPI environment
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);//rank is the process ID
	MPI_Comm_size(MPI_COMM_WORLD, &size);//size is the number of 

	//Input and output file pointers
	FILE *fptr;
	FILE *fptw;
	
	char x,y,z;//File header variables
	char chain_file[30];//File name variable
	char file_number[30];//File index variable
	int fn;//file number
	
	//Counters for chain, bead and comma separated data
	int ccount=0;
	int bcount=0;
	int read=0;
	
	int sindex;//Site index
	int cnum;//Chosen chain
	int bnum;//Chosen bead
	int findex,lindex;//Chosen move direction for terminal beads
	int tavail;//Flag for availability of chosen direction
	//tavail=0 direction is not available
	//tavail=1 direction is available
	int dcount;//Count the number of directions available
	
	/***********************************************************************
	Each move is referred to as bead cycle. 	
	Minimum of 720 bead cycles are required to reconfigure a system.
	There are 2 simulation stages
	Stage 1: Equilibration has 100E6 bead cycles
	Stage 2: Production has 8E6 bead cycles
	***********************************************************************/
	//MC simulation move settings
	long long eqm=100E6;//Number of attempted moves for equilibration
	long long pdm=8E6;//Number of attempted moves for production
	long long maxm=eqm+pdm;//Total number of attempted moves
	long long mcstep=8000;//Frequency with which position files are dumped
	long long sucm,totm;//Move counters
	
	int i,j;//General loop variables

	//Reading data from the input file
	fptr=fopen("mchainsf.csv","r");
	if (fptr== NULL)
	{
		printf("Error reading file\n");
		return 1;
	}
	do{
		//Reads header
		if (bcount==0 && ccount==0)
		{
			read=fscanf(fptr,"%c,%c,%c\n",&x,&y,&z);//File header read
		}	
		//Data reading
		read=fscanf(fptr,"%d,%d,%d\n",&beads[ccount].x[bcount],&beads[ccount].y[bcount],&beads[ccount].z[bcount]);
		if (read==3)
			bcount = bcount+1;
		else
			printf("read error %d\n",read);
		if (bcount==bc)
		{
			ccount=ccount+1;
			bcount=0;
		}
			
	}while (!feof(fptr));
	fclose(fptr);

	srand(time(NULL)+rank*1000);
	seed=(long)(time(NULL)+rank*1000);
	if (seed>=0)
		seed=-1-seed;
	for (i=0;i<100;i++)
		printf("%f %f\n",(double)rand()/RAND_MAX,ran2(&seed));

	//Determining lattice state in the simulation box
	sstate();
	for (sindex=0;sindex<ns;sindex++)
	{
		if (lcs.sz[sindex]==1)
			printf("site: %d %d %d %d\n",sindex,lcs.sx[sindex],lcs.sy[sindex],lcs.sz[sindex]);
		else
			printf("site: %d not occupied\n",sindex);
	}
	//Moves counters
	fn=0;
	sucm=0;
	totm=0;
	
	/**********************************************************************
	MC simulation is performed in 6 steps
	1 Randomly a chain and a bead is picked 
	2 Bead is identified as either terminal or internal
	3 Energy difference is initialized
	4 Site index for a bead is calculated
	5 Trial move is made based on the nature of the bead
	6 Trial move is accepted or rejected based on the metropolis algorithm
	***********************************************************************/

	//MC simulation loop starts here
	do{	
		//Randomly choose a chain	
		cnum = rand()%nc;
		//Randomly choose a bead of the chain
		bnum = rand()%bc;

		/*TERMINAL BEAD MOVES*/
		if(bnum==0)//First bead move
		{
			//Initialize first bead energy difference
			dEf=0.0;
			//Site index of the first bead of chosen chain
			fcalc=beads[cnum].x[0]+lx*beads[cnum].y[0]+lx*ly*beads[cnum].z[0];
			
			tavail=0;//Initialize availability of direction
			dcount=0;//Initialize available directions counter
			//Determine available directions for moving first bead
			//Loop over all directions
			for (i=0;i<6;i++)
			{
				tavail=davail(cnum,i,0);
				if(tavail==1)
					dcount=dcount+1;//Increment direction counter
			}
			if (dcount!=4)
				printf("Error in davail function fbead: %d\n",dcount);

			//Select one of the available directions
			tavail=0;//Reinitialize tavail
			do{
				findex=rand()%6;
				tavail=davail(cnum,findex,0);
			}while(tavail==0);

			//Execute trial move for the first bead
			fmeval(cnum,findex,fcalc);
			if(metrop(dEf)==1)
			{	
				accmov(cnum,0,fcalc,scalc,fpos);
				sucm+=1;
			}
		}			
		
		if(bnum==nb)//Last bead move
		{
			//Initialize last bead energy difference
			dEl=0.0;
			//Site index of the last bead of chosen chain
			lcalc=beads[cnum].x[nb]+lx*beads[cnum].y[nb]+lxly*beads[cnum].z[nb];		
			
			tavail=0;//Initialize availability of direction
			dcount=0;//Initialize available directions counter
			//Determine available directions for moving first bead
			//Loop over all directions
			for (i=0;i<6;i++)
			{
				tavail=davail(cnum,i,1);
				if(tavail==1)
					dcount=dcount+1;//Increment direction counter
			}
			if (dcount!=4)
				printf("Error in davail function\n");

			//Select one of the available directions
			tavail=0;//Reintialize tavail
			do{
				lindex=rand()%6;
				tavail=davail(cnum,lindex,1);
			}while(tavail==0);
		
			//Execute trial move for the last bead
			lmeval(cnum,lindex,lcalc);
			if (metrop(dEl)==1)
			{
				accmov(cnum,nb,lcalc,scalc,npos);
				sucm+=1;
			}
		}

		/*INTERNAL BEAD MOVE*/
		if (bnum!=0 && bnum!=nb)
		{
			dE=0.0;
			pcalc=beads[cnum].x[bnum]+lx*beads[cnum].y[bnum]+lxly*beads[cnum].z[bnum];
			kmoves(cnum,bnum);
			scalc=kpos.ex+lx*kpos.ey+lxly*kpos.ez;
			dE=deltaE(pcalc,scalc);
			if (metrop(dE)==1){	
				accmov(cnum,bnum,pcalc,scalc,kpos);
				sucm+=1;
			}
		}
		totm+=1;

		//Write to file	during production runs
		if(totm>eqm && (totm-eqm)%mcstep==0)
		{
			printf("Writing file initiated\n");
			fn+=1;
			sprintf(chain_file,"echains.%d.%d.csv",rank,fn);
			fptw=fopen(chain_file,"wb");
			fprintf(fptw,"x,y,z\n");
			//Loop over all chains
			for (i=0; i<nc; i++)
			{
				//Loop over beads of a chain
				for (j=0; j<bc; j++)
				{
					fprintf(fptw,"%d,%d,%d\n",beads[i].x[j],beads[i].y[j],beads[i].z[j]);
				}
			}
			fclose(fptw);
			printf("Writing file %d completed\n",fn);
		}

	}while(totm<maxm);

	printf("Total number of moves %lld and successful of moves %lld\n",totm,sucm);
	printf("Simulation complete\n");

	//Verifying results
	//Check if a site has more than one bead
	sstate();
	//Loop over all lattice sites
	for (sindex=0;sindex<ns;sindex++)
	{
		if (lcs.sz[sindex]>1)
		{
			printf("site: %d %d %d %d\n",sindex,lcs.sx[sindex],lcs.sy[sindex],lcs.sz[sindex]);
			printf("Verification Result - Fail\n");
		}
	}

	//Write final position of 36*20=720 beads
    char chain_file_1[30];
	sprintf(chain_file_1,"mchainse.%d.csv",rank);
	fptw = fopen(chain_file_1,"w+");
	fprintf(fptw,"x,y,z\n");
    //Loop over all chains
	for (i=0; i<nc; i++)
	{
		//Loop over beads of a chain
		for (j=0; j<bc; j++)
		{
			fprintf(fptw,"%d,%d,%d\n",beads[i].x[j],beads[i].y[j],beads[i].z[j]);
		}
	}
	fclose(fptw);
    MPI_Finalize();
	return 0;
}
