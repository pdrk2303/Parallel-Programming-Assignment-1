#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <limits>
#include <ctime>
#include <cmath>
#include <chrono>
#include <vector>
using namespace std;
double residual_matrix(double** p, double** a, double** l, double** u, int n) {
	double res = 0.0, res1;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j<n; j++) {
			res1 = 0.0;
			for (int k = 0; k<n; k++) {
				res1 += (p[i][k] * a[k][j]) - (l[i][k] * u[k][j]);
			}
			res += res1*res1;
		}
	}
	return res;
}

double max(double a, double b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

double min(double a, double b) {
    if (a > b) {
        return b;
    } else {
        return a;
    }
}

void Hello(int n, int t) {
    srand(time(NULL));
    
    double **a = (double**)malloc(n * sizeof(double*));  
    double **original_a = (double**)malloc(n * sizeof(double*));  
    for (int i = 0; i < n; ++i) {
        a[i] = (double*)malloc(n * sizeof(double)); 
        original_a[i]=(double*)malloc(n * sizeof(double)); 
        for (int j = 0; j < n; ++j) {
            a[i][j] = drand48()*100;  // Access element using row-major order
            original_a[i][j] = a[i][j];
        }
    }
    double **l = (double**)malloc(n * sizeof(double*)); 
    for (int i = 0; i < n; ++i) {
        l[i] = (double*)malloc(n * sizeof(double)); 
    }
    double **u = (double**)malloc(n * sizeof(double*)); 
    for (int i = 0; i < n; ++i) {
        u[i] = (double*)malloc(n * sizeof(double));
    }
    int* pi = (int*)malloc(n * sizeof(int));
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; ++i) {  
        for (int j = 0; j < n; ++j) {
            if (i > j) {
                u[i][j] = 0;
            } else if (i == j) {
                l[i][j] = 1;
            } else {
                l[i][j] = 0;
            }  // Access element using row-major order
        }
    }

    
    for(int i=0;i<n;i++){ pi[i]=i; }
    
    
    int k_max;
    for (int k = 0; k < n; k++) {
        double max_val = 0;
        // #pragma omp parallel for shared(n) num_threads(t) reduction(max:max_val) reduction(max:k_max)
        for (int k1 = k; k1 < n; k1++) {
            double abs_value = fabs(a[k1][k]);
            if (max_val < abs_value) {
                max_val = abs_value;
                k_max = k1;
            }
        }

        if (max_val == 0) {
            cout << "Error: Singular matrix" << endl;
            return;
        }

        swap(pi[k], pi[k_max]);
        swap(a[k], a[k_max]);   
        for(int i=0;i<k;i++){
            swap(l[k][i],l[k_max][i]);
        }
        u[k][k]=a[k][k];
        double f1 = u[k][k];
        for (int k2 = k + 1; k2 < n; k2++) {
            l[k2][k] = a[k2][k] / f1;
            u[k][k2] = a[k][k2];
        }

        #pragma omp parallel for shared(n) num_threads(t)
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                double l_value = l[i][k];
                double u_value = u[k][j];
                a[i][j] -= l_value * u_value;
            }
        }
    }


    auto end = std::chrono::high_resolution_clock::now();


   


    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken by LU decomposition: " << duration.count() << " milliseconds\n";
    // double** P=(double**)malloc(n * sizeof(double*));
    // for (int i = 0; i < n; ++i) {
    //     P[i] = (double*)malloc(n * sizeof(double)); 
    //     for (int j = 0; j < n; ++j) {
    //         P[i][j] = 0; 
    //     }
    //     P[i][pi[i]]=1;
    // }
    // double norm=residual_matrix(P,original_a,l,u,n);
    // cout<<"Norm is "<<norm<<endl;

    // -----------------------------------------------------------------------------------------------------------------
    for(int i=0;i<n;i++)
        delete[] l[i];
    for(int i=0;i<n;i++)
        delete[] u[i];
    for(int i=0;i<n;i++)
        delete[] a[i];
    delete[] l;
    delete[] u;
    delete[] pi;
    delete[] a;

    return;
}


int main() {
    int n,t;
    cout << "Enter the size of the matrix: ";
    cin >> n;
    cout << "Enter the number of threads: ";
    cin >> t;
    Hello(n, t);
    return 0;
}