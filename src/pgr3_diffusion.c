#include <stdio.h>
#include <math.h>
#include <thrust/host_vector.h>



void calculateDiffusionSerial (double *array, double k, double rt){

}

int main (int argc, char *argv[])
{
  long lengthBar = 0;
  long deltaLength = 0;
  double k = 0;
  double roomTemp = 0;
  thrust::host_vector<double> u;

  // setup/initialize
	if ((argc != 5) || (atoi(argv[2]) > atoi(argv[1]))) {
		cerr << "usage: progName lengthBar deltaLength heatSource roomTemp\n" << endl;
		exit(-1);
	}
	else {
		lengthBar = atol(argv[1]);
		deltaLength = atol(argv[2]);
    k = atof(argv[3]);
    roomTemp = atof(argv[4]);
    //u = (double*)malloc(ceil(lengthBar/(float)deltaLength)*sizeof(double));
    u = thrust::host_vector<double> u(ceil(lengthBar/(float)deltaLength));
    int i;
    for (i = 0; i < ceil(lengthBar/(float)deltaLength); i++){
      u[i] = roomTemp;
    }
	}



   return 0;
}
