#include <vector>

void cpu_matmul_naive(std::vector<float>& C, std::vector<float> const& A, std::vector<float> const& B, int N) 
{
    for(int y=0; y<N; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            float sum = 0;
            for(int k=0; k<N; ++k)
            {
                sum += A[y*N+k] * B[k*N+x];
            }
            C[y*N+x] = sum;
        }
    }
}

void cpu_matmul_improved(std::vector<float>& C, std::vector<float> const& A, std::vector<float> const& B, int N) 
{
    static const int MBS = 8; //block size
    for( int by=0; by<N/MBS; ++by ) //block index 1
    {
        for( int bx=0; bx<N/MBS; ++bx ) //block index 2
        {
            for( int bk=0; bk<N/MBS; ++bk ) //block index 3
            {
                auto y0 = by * MBS;
                auto x0 = bx * MBS;
                auto k0 = bk * MBS;
                for( int y=0; y<MBS; ++y )
                {
                    auto yy = y0 + y;
                    for( int x=0; x<MBS; ++x )
                    {
                        auto xx = x0 + x;
                        float sum = 0.0f;
                        for( int k=0; k<MBS; ++k )
                        {
                            sum += A[yy*N+k0+k] * B[(k0+k)*N+xx];
                        }
                        C[yy*N+xx] += sum;
                    }
                }
            }
        }
    }
}