#include "main.h"
using namespace std;
using namespace chrono;

void deviceInfoPrint();

HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

int main(int argc, char * argv[]) {

	// DEVICE info	
	deviceInfoPrint();

	// Output headers
	SetConsoleTextAttribute(hConsole, 12);
	cout << "\n\n\n" << fixed << setprecision(0);
	cout << "              CPU                         GPU\n\n\n";
	SetConsoleTextAttribute(hConsole, 14);
	cout << "        |     LU      |     LU             QR         Cholesky  |\n";
	cout << "        |-------------|-------------|-------------|-------------|\n";
	

	// RUN for 100
	cout << "   100  |";
	cout << setw(12) << cpuLU(100) << " |";
	cout << setw(12) << gpuLU(100) << " |";
	cout << setw(12) << QR(100) << " |";
	cout << setw(12) << Cholesky(100) << " |\n";

	// RUN for 500
	cout << "   500  |";
	cout << setw(12) <<  cpuLU(500) << " |";
	cout << setw(12) << gpuLU(500) << " |";
	cout << setw(12) << QR(500) << " |";
	cout << setw(12) << Cholesky(500) << " |\n";

	// RUN for 1000
	cout << "  1000  |";
	cout << setw(12) << cpuLU(1000) << " |";
	cout << setw(12) << gpuLU(1000) << " |";
	cout << setw(12) << QR(1000) << " |";
	cout << setw(12) << Cholesky(1000) << " |\n";

	// RUN for 5000
	cout << "  5000  |";
	cout << setw(12) << cpuLU(5000) << " |";
	cout << setw(12) << gpuLU(5000) << " |";
	cout << setw(12) << QR(5000) << " |";
	cout << setw(12) << Cholesky(5000) << " |\n";

	// RUN for 5000
	cout << " 10000  |";
	cout << setw(12) << cpuLU(10000) << " |";
	cout << setw(12) << gpuLU(10000) << " |";
	cout << setw(12) << QR(10000) << " |";
	cout << setw(12) << Cholesky(10000) << " |\n";

	// OUTPUT footer
	cout << "        |-------------------------------------------------------|\n";
	SetConsoleTextAttribute(hConsole, 6);
	cout << "                           Time in MILLISECONDS\n";
	cout << "\n\n";
	SetConsoleTextAttribute(hConsole, 8);
	cout << "done\n\n";

	system("pause");
	return 0;
}


// ------- HELPERS------------------------------------------------------------------


// DEVICE function
void deviceInfoPrint() {
	
	SetConsoleTextAttribute(hConsole, 3);

	// Variables
	int devicesNum;
	int getDevice;
	cudaDeviceProp device;

	// Prep
	cudaGetDeviceCount(&devicesNum);
	cudaGetDevice(&getDevice);
	cudaGetDeviceProperties(&device, getDevice);

	// Variables in print statements
	char arc[8][10] = { "", "Tesla", "Fermi", "Kepler", "Maxwell", "Pascal", "Volta" };
	string name = device.name;
	float clockGhz = float(device.clockRate) / 1000000;
	float memGlob = floor(float(device.totalGlobalMem) / 1000000000 * 100) / 100;
	float memConst = floor(float(device.totalConstMem) / 1000 * 100) / 100;
	int warpSize = device.warpSize;
	int MSnum = device.multiProcessorCount;
	int ntpb = device.maxThreadsPerBlock;
	int numThreadsPerMS = device.maxThreadsPerMultiProcessor;

	cout << "\n\n";
	cout << "\t    ================================================\n";
	cout << "\t                           GPU                      \n";
	cout << "\t    ================================================\n";
	cout << "\t       GPU: " << name << ", clockRate: " << clockGhz << "GHz" << endl;
	cout << "\t       Architecture: " << arc[device.major] << " " << device.major << "." << device.minor << endl;
	cout << "\t       Global Memory: " << memGlob << "GB, constant memory: " << memConst << "KB" << endl;
	cout << "\t       Number of Multiprocessors: " << MSnum << endl;
	cout << "\t       Number of Threads per Block: " << ntpb << endl;
	cout << "\t       Threads per Multiprosessor: " << numThreadsPerMS << endl;
	cout << "\t    -------------------------------------------------\n";
}