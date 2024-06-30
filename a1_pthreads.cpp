#include <bits/stdc++.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <pthread.h>

using namespace std;
using namespace chrono;

typedef struct {
    int num_threads;
    int n;
    double **a;
    int *pi;
    double **l;
    double **u;
} GlobalData;

typedef struct {
    int my_rank;
    int k;
    double max;
    int k_;
    GlobalData *global_data;
} ThreadData;


void* nested_loop_computation(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int thread_id = data->my_rank;
    int num_threads = data->global_data->num_threads;
    int n = data->global_data->n;
    double **a = data->global_data->a;
    int *pi = data->global_data->pi;
    double **l = data->global_data->l;
    double **u = data->global_data->u;
    int k = data->k;
    int start_row = (k+1)+thread_id*(n-(k+1))/num_threads;
    int end_row = (k+1)+(thread_id+1)*(n-(k+1))/num_threads;
    if (thread_id == num_threads - 1) {
        end_row = n;  
    }

    for (int i = start_row; i < end_row; i++) {
        for (int j = k+1; j < n; j++) {
            a[i][j] -= (l[i][k] * u[k][j]);
        }
    }

    pthread_exit(NULL);
}


double residual_matrix(double** p, double** a, double** l, double** u, int n) {
	double c,d;
    c = 0.0;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j<n; j++) {
			d = 0.0;
			for (int k = 0; k<n; k++) {
				d += (p[i][k] * a[k][j]) - (l[i][k] * u[k][j]);
			}
			c += d*d;
		}
	}
	return c;
}


void lu_decomposition(int n, int thread_count, double **a, int *pi, double **l, double **u) {
    int i, j, k, k_;
    double max;

    pthread_t* thread_handles;
    
    ThreadData *thread_data_array[thread_count];
    GlobalData *global_data = new GlobalData();
    global_data->num_threads = thread_count;
    global_data->n = n;
    global_data->a = a;
    global_data->pi = pi;
    global_data->l = l;
    global_data->u = u;

    thread_handles = new pthread_t[thread_count];

    auto start = high_resolution_clock::now();   

    for (i = 0; i < n; i++) {
        pi[i] = i;
        for (j = 0; j < n; j++) {
            l[i][j] = (i == j) ? 1.0 : 0.0;
            u[i][j] = 0.0;
        }
    } 

    for (k = 0; k < n; k++) {
        max = 0.0;
        for (i = k; i < n; i++) {
            if (fabs(a[i][k]) > max) {
                max = fabs(a[i][k]);
                k_ = i;
            }
        }

        if (max == 0) {
            cout << "Error: Singular matrix" << endl;
            exit(1);
        }

        int temp = pi[k];
        pi[k] = pi[k_];
        pi[k_] = temp;
        for (j = 0; j < n; j++) {
            double temp = a[k][j];
            a[k][j] = a[k_][j];
            a[k_][j] = temp;
        }
        for (j = 0; j < k; j++) {
            double temp = l[k][j];
            l[k][j] = l[k_][j];
            l[k_][j] = temp;
        }

        u[k][k] = a[k][k];
        for (i = k + 1; i < n; i++) {
            l[i][k] = a[i][k] / u[k][k];
            u[k][i] = a[k][i];
        }
        
        for (int thread=0; thread < thread_count; thread++) {
            ThreadData *thread_data = new ThreadData();
            thread_data->my_rank = thread;
            thread_data->k = k;
            thread_data->global_data = global_data;
            pthread_create(&thread_handles[thread], NULL, nested_loop_computation, (void*)(thread_data));
        }

        for (int thread = 0; thread < thread_count; thread++) {
            pthread_join(thread_handles[thread], NULL);
        }
    }

    auto end = high_resolution_clock::now();      
    auto time = duration_cast<milliseconds>(end - start).count();
    cout<<"Time taken by LU decomposition: "<< time <<" milliseconds\n"; 

    delete global_data;
    delete[] thread_handles;
}


int main() {
    int n; 
    int thread_count;
    double **a,**tmp, **l, **u,**p;
    int *pi;
    int i, j;


    cout << "Enter the size of the matrix: ";
    cin >> n;
    cout << "Enter the number of threads: ";
    cin >> thread_count;
    
    a = new double*[n];
    tmp = new double*[n];
    l = new double*[n];
    u = new double*[n];
    p = new double*[n];
    pi = new int[n];
    for (i = 0; i < n; i++) {
        a[i] = new double[n];
        p[i] = new double[n];
        tmp[i] = new double[n];
        l[i] = new double[n];
        u[i] = new double[n];
    }

    srand48(time(NULL));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i][j] = drand48(); 
	        tmp[i][j] = a[i][j];
        }
    }

    lu_decomposition(n, thread_count, a, pi, l, u);

    for (int i = 0; i<n; i++) {
		p[i][pi[i]] = 1.0;
	}

	// double norm = residual_matrix(p, tmp, l, u, n);
    // cout << "Value of the L2,1 Norm of the residualis: " << norm << endl;

    for (i = 0; i < n; i++) {
        delete[] a[i];
        delete[] l[i];
        delete[] u[i];
        delete[] tmp[i];
        delete[] p[i];
    }
    delete[] a;
    delete[] l;
    delete[] u;
    delete[] pi;
    delete[] p;
    delete[] tmp;

    return 0;
}

