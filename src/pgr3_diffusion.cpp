#include <stdio.h>
#include <math.h>
#include <iostream>
#include <thrust/host_vector.h>

using namespace std;

#define MIN_VARIATION 0.05

thrust::host_vector< thrust::host_vector<double> > results;

void calculateDiffusionSerial (double k, double rt) {
  bool stopCondition = false;
  long t = 0;
  long i;

  thrust::host_vector<double> prev_u(results[t].size());
  thrust::host_vector<double> array_u(results[t].size());

  while (!stopCondition){
    prev_u = results[t];

    for (i=0; i < prev_u.size(); i++){
      if (i==0){
        array_u[i] = (k + prev_u[i+1])/2;
      }else if (i>0 && i<prev_u.size()-1){
        array_u[i] = (prev_u[i+1] + prev_u[i-1])/2;
      } else {
        array_u[i] = (rt + prev_u[i-1])/2;
      }
    }
    t++;
    results.push_back(array_u);

    bool noVariation = true;
    long j;
    for (j=0; j < array_u.size(); j++) {
      if(array_u[j] <= prev_u[j]+MIN_VARIATION) { //&& array_u[j] >= prev_u[j]-MIN_VARIATION){
        noVariation=false;
      }
    }
    if(noVariation){
      stopCondition = true;
    }
  }
}

int main (int argc, char *argv[]) {
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
    //u = thrust::host_vector<double> u(ceil(lengthBar/(float)deltaLength));
    u.resize(ceil(lengthBar/(float)deltaLength));
    int i;
    for (i = 0; i < ceil(lengthBar/(float)deltaLength); i++){
      u[i] = roomTemp;
    }
    results.push_back(u);
	}

  calculateDiffusionSerial(k, roomTemp);

  printf ("Final length of results:\n");
  printf ("%u ", results.size());
  printf ("\n");

  return 0;
}
