#include <chrono>     // for std::std::chrono::duration_cast
#include <filesystem> // for std::filesystem::path, std::filesystem::directory_iterator, std::filesystem::is_regular_file, std::filesystem::last_write_time, std::filesystem::file_size
#include <iostream>   // for std::cout

namespace fs = std::experimental::filesystem;

int main()
{
    fs::path dir = fs::canonical(".");

    fs::path latest = fs::directory_iterator(dir)->path();
    auto time = fs::last_write_time(latest);
    for (auto& p : fs::directory_iterator(dir))
    {
        auto time2 = fs::last_write_time(p);
        if(time2 > time){ latest = p; time = time2; }
        
        std::cout << p.path().filename();
        std::cout << " " << std::chrono::duration_cast<std::chrono::milliseconds>(decltype(time)::clock::now() - time).count()/1000.0 << " seconds ago.";
        
        if( fs::is_regular_file(p) )
        {
            std::cout << " Size: " << fs::file_size(p) << " bytes.";
        }
        else
        {
            std::cout << " It is a directory.";
        }
        std::cout << "\n";
    }

    std::cout << "latest file: " << latest << "\n";
}