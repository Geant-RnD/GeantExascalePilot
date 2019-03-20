
#include "test_tuple.hpp"

//======================================================================================//

// some short hand definitions
using Generator     = std::mt19937_64;
using A             = ObjectA;
using B             = ObjectB;
using AccessA       = ObjectAccessor<A>;
using AccessB       = ObjectAccessor<B>;
using Aaccess       = ObjectAccessor<A, Generator&>;
using Baccess       = ObjectAccessor<B, Generator&>;
using GeneratorFunc = std::function<void(Generator&)>;

//======================================================================================//

template <typename _Func, typename... _Args>
void exec(_Func&& _func, _Args&&... _args)
{
    _func(std::forward<_Args>(_args)...);
}

//======================================================================================//

void run(uintmax_t nloop, uintmax_t nitr)
{
    // constexpr std::size_t unroll_length = 10;
    // uintmax_t _nitr = nitr / unroll_length;
    // uintmax_t _nmod = nloop % nitr;

    // reference timer
    std::random_device                     rd;
    std::shared_ptr<duration_t>            ref_timer;
    auto                                   seed = rd();
    Generator                              r_generator(seed);
    Generator                              f_generator(seed);
    Generator                              t_generator(seed);
    Generator                              l_generator(seed);
    Generator                              a_generator(seed);
    std::vector<std::pair<double, double>> results;

    // base class object (not derived)
    ObjectA* obj_a = new ObjectA();
    // derived object (with access to base class)
    ObjectB* obj_b    = new ObjectB();
    AccessA* a_access = new AccessA(obj_a);
    AccessB* b_access = new AccessB(obj_b);

    auto record = [&]() {
        results.push_back(std::make_pair(obj_a->GetRandomValue(), obj_b->GetRandomValue()));
        obj_a->Reset();
        obj_b->Reset();
        r_generator = Generator(seed);
        f_generator = Generator(seed);
        t_generator = Generator(seed);
        l_generator = Generator(seed);
        a_generator = Generator(seed);
    };

    //==================================================================================//
    //              Functional section
    //==================================================================================//
    // functional operators
    auto                          b_funct      = [&](Generator& gen) { a_access->generate(gen); };
    auto                          d_funct      = [&](Generator& gen) { b_access->generate(gen); };
    std::array<GeneratorFunc, 10> funct_funct  = { b_funct, d_funct, b_funct, d_funct, b_funct,
                                                  d_funct, b_funct, d_funct, b_funct, d_funct };
    auto                          funct_vector = [&](Generator& gen) {
        for(const auto& itr : funct_funct)
            itr(gen);
    };

    //==================================================================================//
    //          Tuple Accessor section
    //==================================================================================//
    // create tuple of member functions
    auto a_tuple     = [&](Generator& gen) { a_access->generate(gen); };
    auto b_tuple     = [&](Generator& gen) { b_access->generate(gen); };
    auto funct_tuple = [&](Generator& gen) {
        Apply<void>::apply_loop(MakeTuple(a_tuple, b_tuple, a_tuple, b_tuple, a_tuple, b_tuple, a_tuple, b_tuple,
                                          a_tuple, b_tuple),
                                std::ref(gen));
    };

    //==================================================================================//
    //          Lambda Construct section
    //==================================================================================//
    // create tuple of member functions
    auto a_lambda    = [&](Generator& gen) { a_access->generate(l_generator); };
    auto b_lambda    = [&](Generator& gen) { b_access->generate(l_generator); };
    auto funct_lamda = [&](Generator& gen) {
        a_lambda(gen);
        b_lambda(gen);
        a_lambda(gen);
        b_lambda(gen);
        a_lambda(gen);
        b_lambda(gen);
        a_lambda(gen);
        b_lambda(gen);
        a_lambda(gen);
        b_lambda(gen);
    };

    //==================================================================================//
    //              Lambda Accessor section
    //==================================================================================//
    // accessor
    std::function<void(ObjectA&, Generator&)> op_a     = [](ObjectA& m_obj, Generator& gen) { m_obj.generate(gen); };
    std::function<void(ObjectB&, Generator&)> op_b     = [](ObjectB& m_obj, Generator& gen) { m_obj.generate(gen); };
    auto                                      access_a = Aaccess(obj_a, op_a);
    auto                                      access_b = Baccess(obj_b, op_b);
    // create tuple of accessors
    auto funct_access = [&](Generator& gen) {
        Apply<void>::apply_loop(MakeTuple(access_a, access_b, access_a, access_b, access_a, access_b, access_a,
                                          access_b, access_a, access_b),
                                std::ref(gen));
    };

    //==================================================================================//
    //              Function Accessor section
    //==================================================================================//
    // create tuple of accessors
    auto array_access =
        MakeTuple(a_access, b_access, a_access, b_access, a_access, b_access, a_access, b_access, a_access, b_access);
    auto array_funct = MakeTuple(&AccessA::template generate<Generator, A>, &AccessB::template generate<Generator, B>,
                                 &AccessA::template generate<Generator, A>, &AccessB::template generate<Generator, B>,
                                 &AccessA::template generate<Generator, A>, &AccessB::template generate<Generator, B>,
                                 &AccessA::template generate<Generator, A>, &AccessB::template generate<Generator, B>,
                                 &AccessA::template generate<Generator, A>, &AccessB::template generate<Generator, B>);

    auto funct_array = [&](Generator& gen) { Apply<void>::apply_functions(array_access, array_funct, std::ref(gen)); };

    //==================================================================================//
    //          Serial execution section
    //==================================================================================//
    GET_TIMER(ref_s);
    for(uintmax_t i = 0; i < nitr; ++i)
    {
        obj_a->generate(r_generator);
        obj_b->generate(r_generator);
        obj_a->generate(r_generator);
        obj_b->generate(r_generator);
        obj_a->generate(r_generator);
        obj_b->generate(r_generator);
        obj_a->generate(r_generator);
        obj_b->generate(r_generator);
        obj_a->generate(r_generator);
        obj_b->generate(r_generator);
    }
    REPORT_TEST_TIMER(ref_s, "Reference", nloop, nitr, ref_timer);
    record();

    GET_TIMER(funct_s);
    for(uintmax_t i = 0; i < nitr; ++i)
        funct_vector(f_generator);
    REPORT_TEST_TIMER(funct_s, "Functional", nloop, nitr, ref_timer);
    record();

    GET_TIMER(array_a);
    for(uintmax_t i = 0; i < nitr; ++i)
        funct_array(a_generator);
    REPORT_TEST_TIMER(array_a, "Funct Access", nloop, nitr, ref_timer);
    record();

    GET_TIMER(tuple_a);
    for(uintmax_t i = 0; i < nitr; ++i)
        funct_access(a_generator);
    REPORT_TEST_TIMER(tuple_a, "Lambda Access", nloop, nitr, ref_timer);
    record();

    GET_TIMER(tuple_s);
    for(uintmax_t i = 0; i < nitr; ++i)
        funct_tuple(t_generator);
    REPORT_TEST_TIMER(tuple_s, "Tuple", nloop, nitr, ref_timer);
    record();

    GET_TIMER(lambda_s);
    for(uintmax_t i = 0; i < nitr; ++i)
        funct_lamda(l_generator);
    REPORT_TEST_TIMER(lambda_s, "Lambda", nloop, nitr, ref_timer);
    record();

    //==================================================================================//
    //          Report section
    //==================================================================================//
    std::cout << "\n+ Random seed = " << seed << std::endl;
    for(auto& itr : results)
    {
        std::cout << "      " << std::fixed << std::setw(8) << std::setprecision(2) << itr.first << ", " << std::setw(8)
                  << std::setprecision(2) << itr.second << std::endl;
    }

    delete obj_a;
    delete obj_b;
}

//======================================================================================//

int main(int argc, char** argv)
{
    write_app_info(argc, argv);

    uintmax_t nloop = 11;
    uintmax_t nitr  = 1000000;
    uintmax_t nfac  = 2;

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
}

//======================================================================================//
