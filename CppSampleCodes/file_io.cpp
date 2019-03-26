#include <array>
#include <fstream>
#include <iostream>
#include <algorithm>    // std::copy
#include <iterator>     // std::ostream_iterator

struct particle
{
    std::array<double, 3> pos, vel;
    double mass;

    static constexpr const char* delim = "\t";
};

std::ostream& operator<<(std::ostream& s, const particle& p)
{   
    for (const auto& arr : {p.pos, p.vel})
        std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<double>{ s, particle::delim });

    return s << particle::delim << p.mass;
}

std::istream& operator>>(particle& p, std::istream& s)
{
    const auto state = s.rdstate();
    const auto pos = s.tellg();

    std::array<double, 7> temp;

    auto generate_until = [](auto first, auto last, auto gen, auto pred)
    {
        for (bool valid = pred(); first != last && valid; ++first, valid = pred())
        {
            if (valid)
            {
                *first = gen();
            }
            else
            {
                return false;
            }
        }

        return true;
    };

    if (generate_until(temp.begin(), temp.end(),
        [&]() { double val; s >> val; return val; },
        [&]() { return !s.fail(); }))
    {
        return s;
    }
    else
    {
        s.seekg(pos);
        s.setstate(state);

        return s;
    }
}

int main()
{
    return 0;
}
