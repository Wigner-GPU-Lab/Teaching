#ifndef OCTREE_HPP
#define OCTREE_HPP

// STL includes
#include <array>
#include <vector>
#include <stack>
#include <algorithm>
#include <utility>
#include <memory>
#include <iterator>

template <typename T, class Accessor>
class octree
{
private:

	// Member class forward declarations
	//
    class node;
	template <typename TT, typename TTIT, typename TTPTR> class node_iterator;

public:

	// STD typedefs
	//
    typedef T value_type;
    typedef std::size_t size_type;

	// Tree typedefs
	//
    typedef double coord_type;
    typedef double extent_type;
    typedef std::array<coord_type, 3> pos_type;

	// Octree typedefs
	//
	typedef node node_type;
	typedef std::vector<node_type> children_container_type;
	typedef typename children_container_type::iterator children_iterator_type;
	typedef typename children_container_type::const_iterator const_children_iterator_type;
    typedef node_iterator<node_type, children_iterator_type, children_container_type*> iterator;
    typedef node_iterator<const node_type, const_children_iterator_type, const children_container_type*> const_iterator;

    octree() = default;
    octree(const octree& in) = default;
    octree(octree&& in) = default;
    ~octree() = default;

    octree(const pos_type& center, const extent_type& extent, const size_type& threshold) :
        m_center(center),
        m_extent(extent),
        m_threshold(threshold),
        m_node()
	{
		m_node.push_back(node_type(center, extent, static_cast<size_type>(0), threshold));
	}

    // Non-modifying member functions
    //
    const pos_type& center() const { return m_center; }
    const extent_type& extent() const { return m_extent; }
	size_type depth() const { return std::max_element(cbegin(), cend(), [](const node_type& node1, const node_type& node2) { return node1.level() < node2.level(); })->level(); }
	size_type size() const { return std::accumulate(cbegin(), cend(), static_cast<size_type>(0)); }
	size_type nodes() const { return std::distance(cbegin(), cend()); }

    // Non-modifying iterator interface
    //
    const_iterator cbegin() const
	{
		typename const_iterator::trace_type trace;

		trace.emplace(m_node.cbegin(), &m_node);

		while (!trace.top().first->am_i_leaf())
			trace.emplace(trace.top().first->m_children.cbegin(), &trace.top().first->m_children);

		return trace;
	}
    const_iterator cend() const
	{
		typename const_iterator::trace_type trace;

		trace.emplace(m_node.cend(), &m_node);

		return trace;
	}

    // Modifying member functions
    //
    void insert(const value_type& value)
    {
        m_node.at(0).insert(value);
    }

    // Modifying iterator interface
    //
    iterator begin()
	{
		typename iterator::trace_type trace;
		
		trace.emplace(m_node.begin(), &m_node);

		while (!trace.top().first->am_i_leaf())
			trace.push(std::make_pair(trace.top().first->m_children.begin(), &trace.top().first->m_children));

		return trace;
	}
    iterator end()
	{
		typename iterator::trace_type trace;

		trace.emplace(m_node.end(), &m_node);

		return trace;
	}

private:

    pos_type m_center;
    extent_type m_extent;
    size_type m_threshold;
    children_container_type m_node;

    class node
    {
		friend class octree<T, Accessor>;
                template <typename TT, typename TTIT, typename TTPTR> friend class node_iterator;
                //template<> friend class node_iterator<node_type, children_iterator_type, children_container_type*>;
                //template<> friend class node_iterator<const node_type, const_children_iterator_type, const children_container_type*>;
		
                // Does not compile under gcc 4.8.2
                //friend class iterator;
                //friend class const_iterator;

    public:

		// Value typedefs
		//
        typedef std::vector<value_type> value_container_type;
		typedef typename value_container_type::size_type size_type;

		typedef typename value_container_type::iterator iterator;
		typedef typename value_container_type::const_iterator const_iterator;
		typedef typename value_container_type::reverse_iterator reverse_iterator;
		typedef typename value_container_type::const_reverse_iterator const_reverse_iterator;

        node() = default;
        node(const node& in) = default;
        node(node&& in) = default;
        ~node() = default;

		// Constructor that node calls on subdivision
        node(const pos_type& center, const extent_type& extent, const size_type& level, const size_type& threshold) :
            m_data(),
            m_center(center),
            m_extent(extent),
            m_level(level),
			m_threshold(threshold),
			m_children(0) {}

        node& operator=(const node& in)
        {
            m_data = in.m_data;
            m_center = in.m_center;
            m_extent = in.m_extent;
            m_level = in.m_level;
			m_threshold = in.m_threshold;
			m_children = in.m_children;
            return *this;
        }

        // Cast operator for octree::size() (not nice, but ok for private member class)
        //
        operator size_type() const { return size(); }

        // Non-modifying member functions
        //
        const pos_type& center() const { return m_center; }
        const extent_type& extent() const { return m_extent; }
        size_type level() const { return m_level; }
        size_type size() const { return m_data.size(); }
		bool empty() const { return m_data.empty(); }
        bool contains(const pos_type& pos) const
        {
            return
                (std::abs(m_center.at(0) - pos.at(0)) <= m_extent) &&
                (std::abs(m_center.at(1) - pos.at(1)) <= m_extent) &&
                (std::abs(m_center.at(2) - pos.at(2)) <= m_extent);
        }

        // Non-modifying iterator interface
        //
        const_iterator cbegin() const { return m_data.cbegin(); }
        const_iterator cend() const { return m_data.cend(); }
        const_reverse_iterator crbegin() const { return m_data.crbegin(); }
        const_reverse_iterator crend() const { return m_data.crend(); }

        // Modifying member functions
        //
        void insert(const value_type& value)
        {
			if (contains(Accessor()(value)))
			{
				if (am_i_leaf()) // If I am a leaf node that could contain the value then
				{
					if (size() < m_threshold) // If the value fits into the node, insert it
					{
						m_data.push_back(value);
					}
					else // Otherwise spawn new child nodes and insert
					{
						subdivide();
						child_insert(value);
					}
				}
				else // Otherwise find the child that contains the position
				{
					child_insert(value);
				}
			}
			else
				throw std::out_of_range("Position is out of tree extent");
        }

        // Modifying iterator interface
        //
        iterator begin() { return m_data.begin(); }
        iterator end() { return m_data.end(); }
        reverse_iterator rbegin() { return m_data.rbegin(); }
        reverse_iterator rend() { return m_data.rend(); }

    private:

        void subdivide()
        {
            // Calculate new node properties
            extent_type new_extent = this->extent() / 2;
            size_type new_level = this->level() + 1;
            std::array<pos_type, 8> new_centers
            { {
                { this->center().at(0) - new_extent, this->center().at(1) - new_extent, this->center().at(2) - new_extent },
                { this->center().at(0) + new_extent, this->center().at(1) - new_extent, this->center().at(2) - new_extent },
                { this->center().at(0) - new_extent, this->center().at(1) + new_extent, this->center().at(2) - new_extent },
                { this->center().at(0) + new_extent, this->center().at(1) + new_extent, this->center().at(2) - new_extent },
                { this->center().at(0) - new_extent, this->center().at(1) - new_extent, this->center().at(2) + new_extent },
                { this->center().at(0) + new_extent, this->center().at(1) - new_extent, this->center().at(2) + new_extent },
                { this->center().at(0) - new_extent, this->center().at(1) + new_extent, this->center().at(2) + new_extent },
                { this->center().at(0) + new_extent, this->center().at(1) + new_extent, this->center().at(2) + new_extent }
                } };

            // Instantiate new node instances
			m_children.reserve(new_centers.size());
			for (size_type i = 0; i < new_centers.size(); ++i)
				m_children.push_back(node_type(new_centers.at(i), new_extent, new_level, m_threshold));

            // Populate new nodes with contents of the old
            for (const auto& entry : m_data)
                insert(entry);

			// Clear old data
			m_data.clear();
        }

		void child_insert(const value_type& value)
		{
			auto it = std::find_if(m_children.begin(), m_children.end(), [&value](const node_type& child) { return child.contains(Accessor()(value)); });

			if (it != m_children.end())
				it->insert(value);
			else
				throw "This should never have happened.";
		}

        bool am_i_leaf() const { return m_children.size() == 0; }

        value_container_type m_data;

        pos_type m_center;
        extent_type m_extent;
        size_type m_level;
		size_type m_threshold;

		children_container_type m_children;
    };

	template <typename TT, typename TTIT, typename TTPTR>
	class node_iterator : public std::iterator<std::forward_iterator_tag, TT>
	{
		template <typename TT2, typename TTIT2, typename TTPTR2>
		friend class node_iterator;

	public:

                // ISO C++ requires template-dependent typedefs not to be inferenced
                typedef typename std::iterator<std::forward_iterator_tag, TT>::value_type value_type;
                typedef typename std::iterator<std::forward_iterator_tag, TT>::difference_type difference_type;
                typedef typename std::iterator<std::forward_iterator_tag, TT>::pointer pointer;
                typedef typename std::iterator<std::forward_iterator_tag, TT>::reference reference;
                typedef typename std::iterator<std::forward_iterator_tag, TT>::iterator_category iterator_category;

		typedef std::stack<std::pair<TTIT, TTPTR>> trace_type;

		node_iterator() = default;
		node_iterator(const node_iterator& in) = default;
		node_iterator(node_iterator&& in) = default;
		~node_iterator() = default;

		node_iterator(const trace_type& trace) : m_trace(trace) {}

		node_iterator& operator=(const node_iterator& in)
		{
			m_trace = in.m_trace;

			return *this;
		}

		// EqualityComparable concept
		//
		template <typename TT2, typename TTIT2, typename TTPTR2>
		bool operator==(const node_iterator<TT2, TTIT2, TTPTR2>& rhs) const
		{
			if (m_trace.top().second == rhs.m_trace.top().second)
				return m_trace.top().first == rhs.m_trace.top().first;
			else
				return false;
		}

		// Iterator concept
		//
		node_iterator operator++()
		{
			climb_up_while_i_am_last_node<std::is_const<TT>::value>();

			side_step();

			shoot_down_as_deep_as_possible<std::is_const<TT>::value>();

			return *this;
		}
		reference operator*() { return *m_trace.top().first; }

		// InputIterator concept
		//
		node_iterator operator++(int)
		{
			node_iterator result(*this);

			++(*this);

			return result;
		}
		template <typename TT2, typename TTIT2, typename TTPTR2>
		bool operator!=(const node_iterator<TT2, TTIT2, TTPTR2>& rhs) const
		{
			if (m_trace.top().second == rhs.m_trace.top().second)
				return m_trace.top().first != rhs.m_trace.top().first;
			else
				return true;
		}
		pointer operator->() const { return &(*m_trace.top().first); }

    private:

	// Member function template specialization trick from http://stackoverflow.com/questions/3052579/explicit-specialization-in-non-namespace-scope
        template <bool B>
        struct identity { static const bool value = B; };

        template <bool Const>
        void climb_up_while_i_am_last_node()
        {
            climb_up_while_i_am_last_node_impl(identity<Const>());
        };

        //template <bool>
        void climb_up_while_i_am_last_node_impl(identity<false>)
	{
		while ((m_trace.top().first + 1) == m_trace.top().second->end()) // While the top of the trace points to a last child
		{
			if (m_trace.size() != 1) // and it does not correspond to the root node,
				m_trace.pop(); // climb up the tree
			else
				break; // otherwise stop climbing
		}
	}
	void climb_up_while_i_am_last_node_impl(identity<true>)
	{
		while ((m_trace.top().first + 1) == m_trace.top().second->cend()) // While the top of the trace points to a last child
		{
			if (m_trace.size() != 1) // and it does not correspond to the root node,
				m_trace.pop(); // climb up the tree
			else
				break; // otherwise stop climbing
		}
	}

        void side_step()
        {
            ++m_trace.top().first;
        }

        template <bool Const>
	void shoot_down_as_deep_as_possible()
        {
            shoot_down_as_deep_as_possible_impl(identity<Const>());
        }

        //template <bool>
        void shoot_down_as_deep_as_possible_impl(identity<false>)
        {
            if (m_trace.size() != 1)
		{
			while (!m_trace.top().first->am_i_leaf())
				m_trace.push(std::make_pair(m_trace.top().first->m_children.begin(), &m_trace.top().first->m_children));
		}
	}
	void shoot_down_as_deep_as_possible_impl(identity<true>)
	{
		if (m_trace.size() != 1)
		{
			while (!m_trace.top().first->am_i_leaf())
				m_trace.emplace(m_trace.top().first->m_children.cbegin(), &m_trace.top().first->m_children);
		}
	}

        trace_type m_trace;
    };
};

#endif // OCTREE_HPP
