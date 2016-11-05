#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <iostream> 

using namespace std;

thrust::host_vector< thrust::host_vector<double> > times;

void calculateDiffusionSerial (thrust::host_vector<double> u, double k, double rt){
  times.push_back(u);
  bool stop = false;
  long t = 0;
  thrust::host_vector<double> new_u(u.size());
  while (!stop){
     u = times[t];
     int i;
     for(i = 0; i < u.size(); i++){
       if(i==0){
         new_u[i] = (k+u[i+1])/2;
       }else if(i < u.size()-1){
         new_u[i] = (u[i-1] + u[i+1]) /2;
       }else{
         new_u[i] = (u[i-1]+rt)/2;
       }
     }
     times.push_back(new_u);
     t++;
     bool allTempsEqual=true;
     long j;
     for (j=0; j < new_u.size(); j++){
       if(u[j] != new_u[j]){
         allTempsEqual=false;
       }
     }
     if(allTempsEqual){
        stop = true;
     }
  }
}

int main(int argc, char *argv[]) 
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
  } else {
    lengthBar = atol(argv[1]);
    deltaLength = atol(argv[2]);
    k = atof(argv[3]);
    roomTemp = atof(argv[4]);
    //u = (double*)malloc(ceil(lengthBar/(float)deltaLength)*sizeof(double));
    u.resize(ceil(lengthBar/(float)deltaLength));
    int i;
    for (i = 0; i < ceil(lengthBar/(float)deltaLength); i++){
      u[i] = roomTemp;
    }
    cout << "lenghtBar: " << lengthBar << " deltaLenght: " << deltaLength << " K: " << k << " Room Temp: " << roomTemp << " u Size: " << u.size() << " u[0]: " << u[0]<< endl;
    cout << "initiating serial" << endl;
    calculateDiffusionSerial(u, k, roomTemp);
    cout << "finished!!! Times to diffuse:  " << times.size() << endl; 
  }
  return 0; 
}
