

// without OpenACC
void compute(double *a, double *b, double *c, int n)
{
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// with OpenACC
void compute(double *a, double *b, double *c, int n)
{
    #pragma acc parallel loop
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void a()
{
    #pragma acc parallel
    {
        #pragma acc loop
                for (i = 0; i < N; i++)
                {
                    C[i] = A[i] + B[i];
                }
        #pragma acc loop
                for (i = 0; i < N; i++)
                {
                    D[i] = C[i] * A[i];
                }
    }
}


// Here, the parallel region starts in gang-redundant mode. When the loop directive is encountered, 
// execution switches to gang-partitioned mode because of the gang clause.
void b()
{
    #pragma acc parallel
    {
       int b = a + c;
    #pragma acc loop gang
       for (i = 0; i < N; i++) {
           ...
       }
    }   
}