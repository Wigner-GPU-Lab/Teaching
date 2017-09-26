double sq(double x){ return x*x; }

__kernel void sssh(__global double* PsiR, __global double* PsiI, __global double* U, __global float4* A, double dx, int n, int idx)
{
    double h = 0.0008;
    double m = 60.0;

    int i = get_global_id(0);
    double dr, di;
    double dxdx = sq(dx);
    int read = idx*n;
    int write = (1-idx)*n;
    if(i==0)
    {
        dr = (PsiR[read+0] - 2.0*PsiR[read+1] + PsiR[read+2])/dxdx;
        di = (PsiI[read+0] - 2.0*PsiI[read+1] + PsiI[read+2])/dxdx;
    }
    else if(i==n-1)
    {
        dr = (PsiR[read+n-3] - 2.0*PsiR[read+n-2] + PsiR[read+n-1])/dxdx;
        di = (PsiI[read+n-3] - 2.0*PsiI[read+n-2] + PsiI[read+n-1])/dxdx;
    }
    else
    {
        dr = (PsiR[read+i-1] - 2.0*PsiR[read+i] + PsiR[read+i+1])/dxdx;
        di = (PsiI[read+i-1] - 2.0*PsiI[read+i] + PsiI[read+i+1])/dxdx;
    }

    PsiR[write+i] = PsiR[read+i] + h * (U[i] * PsiI[read+i] - di / (2.0*m));
    PsiI[write+i] = PsiI[read+i] + h * (dr / (2.0*m) - U[i] * PsiR[read+i]);

    float sz = (float)get_global_size(0);
    float4 l;
    l.x = (i-sz/2.0f)/sz;
    l.y = -0.75f + 0.25f*(float)sqrt( sq(PsiR[write+i]) + sq(PsiI[write+i]));
    A[i] = l;
}
