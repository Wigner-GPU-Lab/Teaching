#include <utility>

template <typename T>
struct double_buffer
{
    T front, back;

    void swap()
    {
        std::swap(front, back);
    }
};
