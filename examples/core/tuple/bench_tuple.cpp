
#include "test_tuple.hpp"

//======================================================================================//

// some short hand definitions
using A       = ObjectA;
using B       = ObjectB;
using AccessA = ObjectAccessor<A>;
using AccessB = ObjectAccessor<B>;

static uintmax_t                              nfac = 2;
static std::random_device                     rd;
static auto                                   seed = rd();
static Generator                              generator(seed);
static std::vector<std::pair<double, double>> results;
static ObjectA*                               obj_a  = new ObjectA();
static ObjectB*                               obj_b  = new ObjectB();
static VirtualA*                              virt_a = new VirtualA();
static VirtualB*                              virt_b = new VirtualB();
void                                          record()
{
    results.push_back(std::make_pair(obj_a->GetRandomValue() + virt_a->GetRandomValue(),
                                     obj_b->GetRandomValue() + virt_b->GetRandomValue()));
    obj_a->Reset();
    obj_b->Reset();
    virt_a->Reset();
    virt_b->Reset();
    generator = Generator(seed);
}

//======================================================================================//

class Executor
{
public:
    Executor(std::string label, uintmax_t nloop, uintmax_t nitr, Generator* gen)
    : m_nloop(nloop)
    , m_nitr(nitr)
    , m_label(label)
    , m_generator(gen)
    {
    }

    template <typename _Funct>
    Executor(std::string label, uintmax_t nloop, uintmax_t nitr, _Funct&& func, Generator* gen)
    : m_nloop(nloop)
    , m_nitr(nitr)
    , m_label(label)
    , m_exec(std::forward<_Funct>(func))
    , m_generator(gen)
    {
    }

    template <typename _Funct>
    void SetExec(_Funct&& func)
    {
        m_exec = std::forward<_Funct>(func);
    }

    void SetGenerator(Generator& gen) { m_generator = &gen; }

    void exec()
    {
        GET_TIMER(timer);
        for(uintmax_t i = 0; i < m_nitr; ++i)
        {
            m_exec(*m_generator);
        }
        REPORT_TEST_TIMER(timer, m_label.c_str(), m_nloop, m_nitr, ref_timer());
    }

    void exec_virtual(std::vector<ObjectA*>& virtual_vector)
    {
        GET_TIMER(timer);
        for(uintmax_t i = 0; i < m_nitr; ++i)
        {
            for(const auto& itr : virtual_vector)
                itr->generate(*m_generator);
        }
        REPORT_TEST_TIMER(timer, m_label.c_str(), m_nloop, m_nitr, ref_timer());
    }

private:
    uintmax_t                       m_nloop;
    uintmax_t                       m_nitr;
    std::string                     m_label;
    std::function<void(Generator&)> m_exec      = [](Generator&) {};
    Generator*                      m_generator = nullptr;
};

//======================================================================================//

void run_reference(uintmax_t nloop, uintmax_t nitr)
{
    auto func = [&](Generator& gen) {
        for(int i = 0; i < 10; ++i)
        {
            if(i % 2 == 0)
                obj_a->generate(gen);
            else
                obj_b->generate(gen);
        }
    };

    Executor executor("Reference", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}
//======================================================================================//

void run_functional(uintmax_t nloop, uintmax_t nitr)
{
    auto func = [&](Generator& gen) {
        using GeneratorFunc                       = std::function<void()>;
        auto                          b_funct     = [&]() { AccessA(obj_a).generate(gen); };
        auto                          d_funct     = [&]() { AccessB(obj_b).generate(gen); };
        std::array<GeneratorFunc, 10> funct_funct = { b_funct, d_funct, b_funct, d_funct, b_funct,
                                                      d_funct, b_funct, d_funct, b_funct, d_funct };
        for(const auto& itr : funct_funct)
            itr();
    };

    Executor executor("Functional", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}

//======================================================================================//

void run_virtual_class(uintmax_t nloop, uintmax_t nitr)
{
    std::vector<ObjectA*> virtual_vector(10, nullptr);
    for(uint i = 0; i < 10; i += 2)
        virtual_vector[i] = virt_a;
    for(uint i = 1; i < 10; i += 2)
        virtual_vector[i] = virt_b;

    Executor executor("Virtual (class)", nloop, nitr, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec_virtual(virtual_vector);
    record();
}

//======================================================================================//

void run_virtual_lambda(uintmax_t nloop, uintmax_t nitr)
{
    std::vector<ObjectA*> virtual_vector(10, nullptr);
    for(uint i = 0; i < 10; i += 2)
        virtual_vector[i] = virt_a;
    for(uint i = 1; i < 10; i += 2)
        virtual_vector[i] = virt_b;

    auto func = [&](Generator& gen) {
        for(const auto& itr : virtual_vector)
            itr->generate(gen);
    };

    Executor executor("Virtual (lambda)", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}

//======================================================================================//

void run_function_access(uintmax_t nloop, uintmax_t nitr)
{
    // create tuple of accessors
    auto array_access = MakeTuple(AccessA(obj_a), AccessB(obj_b), AccessA(obj_a), AccessB(obj_b), AccessA(obj_a),
                                  AccessB(obj_b), AccessA(obj_a), AccessB(obj_b), AccessA(obj_a), AccessB(obj_b));
    auto array_funct =
        MakeTuple(&AccessA::template generate<A>, &AccessB::template generate<B>, &AccessA::template generate<A>,
                  &AccessB::template generate<B>, &AccessA::template generate<A>, &AccessB::template generate<B>,
                  &AccessA::template generate<A>, &AccessB::template generate<B>, &AccessA::template generate<A>,
                  &AccessB::template generate<B>);

    auto func = [&](Generator& gen) { Apply<void>::apply_functions(array_access, array_funct, std::ref(gen)); };

    Executor executor("Funct Access", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}

//======================================================================================//

void run_ctors_base(uintmax_t nloop, uintmax_t nitr)
{
    auto func = [&](Generator& gen) {
        auto object_tuple = MakeTuple(obj_a, obj_b, obj_a, obj_b, obj_a, obj_b, obj_a, obj_b, obj_a, obj_b);
        typedef Tuple<ConstructAccessor, ConstructAccessor, ConstructAccessor, ConstructAccessor, ConstructAccessor,
                      ConstructAccessor, ConstructAccessor, ConstructAccessor, ConstructAccessor, ConstructAccessor>
            ctors_tuple;
        Apply<void>::apply_access<ctors_tuple, decltype(object_tuple), Generator&>(object_tuple, std::ref(gen));
    };

    Executor executor("Ctors Base", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}

//======================================================================================//

void run_ctors_virtual(uintmax_t nloop, uintmax_t nitr)
{
    auto func = [&](Generator& gen) {
        auto object_tuple = MakeTuple(virt_a, virt_b, virt_a, virt_b, virt_a, virt_b, virt_a, virt_b, virt_a, virt_b);
        typedef Tuple<ConstructAccessor, ConstructAccessor, ConstructAccessor, ConstructAccessor, ConstructAccessor,
                      ConstructAccessor, ConstructAccessor, ConstructAccessor, ConstructAccessor, ConstructAccessor>
            ctors_tuple;
        Apply<void>::apply_access<ctors_tuple, decltype(object_tuple), Generator&>(object_tuple, std::ref(gen));
    };

    Executor executor("Ctors Virtual", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}

//======================================================================================//

void run_ctors_base_ab(uintmax_t nloop, uintmax_t nitr)
{
    auto func = [&](Generator& gen) {
        auto object_tuple = MakeTuple(obj_a, obj_b, obj_a, obj_b, obj_a, obj_b, obj_a, obj_b, obj_a, obj_b);
        typedef Tuple<ConstructAccessorA, ConstructAccessorB, ConstructAccessorA, ConstructAccessorB,
                      ConstructAccessorA, ConstructAccessorB, ConstructAccessorA, ConstructAccessorB,
                      ConstructAccessorA, ConstructAccessorB>
            ctors_tuple;
        Apply<void>::apply_access<ctors_tuple, decltype(object_tuple), Generator&>(object_tuple, std::ref(gen));
    };

    Executor executor("Ctors Base AB", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}

//======================================================================================//

void run_ctors_virtual_ab(uintmax_t nloop, uintmax_t nitr)
{
    auto func = [&](Generator& gen) {
        auto object_tuple = MakeTuple(virt_a, virt_b, virt_a, virt_b, virt_a, virt_b, virt_a, virt_b, virt_a, virt_b);
        typedef Tuple<ConstructAccessorA, ConstructAccessorB, ConstructAccessorA, ConstructAccessorB,
                      ConstructAccessorA, ConstructAccessorB, ConstructAccessorA, ConstructAccessorB,
                      ConstructAccessorA, ConstructAccessorB>
            ctors_tuple;
        Apply<void>::apply_access<ctors_tuple, decltype(object_tuple), Generator&>(object_tuple, std::ref(gen));
    };

    Executor executor("Ctors Virtual AB", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}

//======================================================================================//

void run_tuple(uintmax_t nloop, uintmax_t nitr)
{
    // create tuple of member functions
    auto func = [&](Generator& gen) {
        auto a_tuple = [&]() { AccessA(obj_a).generate(gen); };
        auto b_tuple = [&]() { AccessB(obj_b).generate(gen); };
        Apply<void>::apply_loop(
            MakeTuple(a_tuple, b_tuple, a_tuple, b_tuple, a_tuple, b_tuple, a_tuple, b_tuple, a_tuple, b_tuple));
    };

    Executor executor("Tuple", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}

//======================================================================================//

void run_lambda(uintmax_t nloop, uintmax_t nitr)
{
    // create tuple of member functions
    auto func = [&](Generator& gen) {
        auto a_lambda = [&]() { AccessA(obj_a).generate(gen); };
        auto b_lambda = [&]() { AccessB(obj_b).generate(gen); };
        for(int i = 0; i < 10; ++i)
        {
            if(i % 2 == 0)
                a_lambda();
            else
                b_lambda();
        }
    };

    Executor executor("Lambda", nloop, nitr, func, &generator);
    TIMEMORY_BASIC_AUTO_TIMER();
    executor.exec();
    record();
}

//======================================================================================//

void run(uintmax_t nloop, uintmax_t nitr)
{
    {
        TIMEMORY_BASIC_AUTO_TIMER();
        run_reference(nloop, nitr);
        run_functional(nloop, nitr);
        run_virtual_class(nloop, nitr);
        run_lambda(nloop, nitr);
        run_tuple(nloop, nitr);
        run_function_access(nloop, nitr);
        run_virtual_lambda(nloop, nitr);
        run_ctors_base(nloop, nitr);
        run_ctors_base_ab(nloop, nitr);
        run_ctors_virtual(nloop, nitr);
        run_ctors_virtual_ab(nloop, nitr);
    }

    std::cout << "\n+ Random seed = " << seed << std::endl;
    for(auto& itr : results)
    {
        std::cout << "      " << std::fixed << std::setw(8) << std::setprecision(2) << itr.first << ", " << std::setw(8)
                  << std::setprecision(2) << itr.second << std::endl;
    }
    results.clear();
    ref_timer().reset();
}

//======================================================================================//

int main(int argc, char** argv)
{
    write_app_info(argc, argv);

#if defined(GEANT_USE_TIMEMORY)
    auto manager = tim::manager::instance();
#endif

    typedef usage_tuple<usage::peak_rss, usage::current_rss, usage::stack_rss, usage::data_rss, usage::num_swap,
                        usage::num_io_in, usage::num_io_out, usage::num_minor_page_faults, usage::num_major_page_faults>
        usage_tuple_t;

    usage_tuple_t initial_usage;
    usage_tuple_t final_usage;
    initial_usage.record();

    uintmax_t nloop = 11;
    uintmax_t nitr  = 1000000;
    if(argc > 1)
        nloop = static_cast<uintmax_t>(atol(argv[1]));
    if(argc > 2)
        nfac = static_cast<uintmax_t>(atol(argv[2]));
    if(argc > 3)
        nitr = static_cast<uintmax_t>(atol(argv[3]));

    for(uintmax_t i = 0; i < nloop; ++i)
    {
        std::cerr << std::endl;
        run(i, nitr * ((i > 0) ? i * nfac : 1));
    }
    std::cerr << std::endl;

    final_usage.record();
    std::cout << "initial: \n" << initial_usage << std::endl;
    std::cout << "final: \n" << final_usage << std::endl;

#if defined(GEANT_USE_TIMEMORY)
    manager->report();
#endif
}

//======================================================================================//
