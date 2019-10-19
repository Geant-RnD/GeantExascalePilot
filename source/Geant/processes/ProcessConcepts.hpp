//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/processes/ProcessConcepts.hpp
 * @brief Memory pool for device and host allocations.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Logger.hpp"
#include "Geant/core/Tuple.hpp"

#include <timemory/timemory.hpp>

#include <functional>
#include <string>
#include <type_traits>

namespace geantx
{
struct Process;
namespace mpl
{
//--------------------------------------------------------------------------------------//
//
//                              AT-REST
//
//--------------------------------------------------------------------------------------//

template <typename ProcessType, typename ParticleType>
struct AtRest
{
    static_assert(std::is_base_of<Process, ProcessType>::value,
                  "ProcessType must derive for Process");

    //----------------------------------------------------------------------------------//
    //                                  GPIL
    //----------------------------------------------------------------------------------//

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnableAtRestGPIL), int>               = 0,
              std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
    static double GPIL(_Args&&... args)
    {
        ProcessType p;
        return p.AtRestGPIL(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnableAtRestGPIL), int>              = 0,
              std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
    static double GPIL(_Args&&... args)
    {
        ProcessType p;
        return p.template AtRestGPIL<ParticleType>(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<!(_Proc::EnableAtRestGPIL), int> = 0>
    static double GPIL(_Args&&...)
    {
        return std::numeric_limits<double>::max();
    }

    //----------------------------------------------------------------------------------//
    //                                  DO-IT
    //----------------------------------------------------------------------------------//

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnableAtRestDoIt), int>               = 0,
              std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
    static void DoIt(_Args&&... args)
    {
        ProcessType p;
        p.AtRestDoIt(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnableAtRestDoIt), int>              = 0,
              std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
    static void DoIt(_Args&&... args)
    {
        ProcessType p;
        p.template AtRestDoIt<ParticleType>(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<!(_Proc::EnableAtRestDoIt), int> = 0>
    static void DoIt(_Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//
//
//                              Along-Step
//
//--------------------------------------------------------------------------------------//

template <typename ProcessType, typename ParticleType>
struct AlongStep
{
    static_assert(std::is_base_of<Process, ProcessType>::value,
                  "ProcessType must derive for Process");

    //----------------------------------------------------------------------------------//
    //                                  GPIL
    //----------------------------------------------------------------------------------//

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnableAlongStepGPIL), int>            = 0,
              std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
    static double GPIL(_Args&&... args)
    {
        geantx::Log(kInfo) << GEANT_HERE << "along step generic";
        ProcessType p;
        return p.AlongStepGPIL(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnableAlongStepGPIL), int>           = 0,
              std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
    static double GPIL(_Args&&... args)
    {
        geantx::Log(kInfo) << GEANT_HERE << "along step specific";
        ProcessType p;
        return p.template AlongStepGPIL<ParticleType>(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<!(_Proc::EnableAlongStepGPIL), int> = 0>
    static double GPIL(_Args&&...)
    {
        geantx::Log(kInfo) << GEANT_HERE << "along step disable";
        return std::numeric_limits<double>::max();
    }

    //----------------------------------------------------------------------------------//
    //                                  DO-IT
    //----------------------------------------------------------------------------------//

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnableAlongStepDoIt), int>            = 0,
              std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
    static void DoIt(_Args&&... args)
    {
        geantx::Log(kInfo) << GEANT_HERE << "along step generic";
        ProcessType p;
        p.AlongStepDoIt(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnableAlongStepDoIt), int>           = 0,
              std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
    static void DoIt(_Args&&... args)
    {
        geantx::Log(kInfo) << GEANT_HERE << "along step specific";
        ProcessType p;
        p.template AlongStepDoIt<ParticleType>(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<!(_Proc::EnableAlongStepDoIt), int> = 0>
    static void DoIt(_Args&&...)
    {
        geantx::Log(kInfo) << GEANT_HERE << "along step disable";
        std::cout << tim::demangle(typeid(Specialized).name()) << std::endl;
    }
};

//--------------------------------------------------------------------------------------//
//
//                              Post-Step
//
//--------------------------------------------------------------------------------------//

template <typename ProcessType, typename ParticleType>
struct PostStep
{
    static_assert(std::is_base_of<Process, ProcessType>::value,
                  "ProcessType must derive for Process");

    //----------------------------------------------------------------------------------//
    //                                  GPIL
    //----------------------------------------------------------------------------------//

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnablePostStepGPIL), int>             = 0,
              std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
    static double GPIL(_Args&&... args)
    {
        geantx::Log(kInfo) << GEANT_HERE << "post step generic";
        ProcessType p;
        return p.PostStepGPIL(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnablePostStepGPIL), int>            = 0,
              std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
    static double GPIL(_Args&&... args)
    {
        geantx::Log(kInfo) << GEANT_HERE << "post step specific";
        ProcessType p;
        return p.template PostStepGPIL<ParticleType>(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<!(_Proc::EnablePostStepGPIL), int> = 0>
    static double GPIL(_Args&&...)
    {
        geantx::Log(kInfo) << GEANT_HERE << "post step disabled";
        return std::numeric_limits<double>::max();
    }

    //----------------------------------------------------------------------------------//
    //                                  DO-IT
    //----------------------------------------------------------------------------------//

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnablePostStepDoIt), int>             = 0,
              std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
    static void DoIt(_Args&&... args)
    {
        geantx::Log(kInfo) << GEANT_HERE << "post step generic";
        ProcessType p;
        p.PostStepDoIt(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<(_Proc::EnablePostStepDoIt), int>            = 0,
              std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
    static void DoIt(_Args&&... args)
    {
        geantx::Log(kInfo) << GEANT_HERE << "post step specific";
        ProcessType p;
        p.template PostStepDoIt<ParticleType>(std::forward<_Args>(args)...);
    }

    template <typename _Proc = ProcessType, typename _Part = ParticleType,
              typename... _Args, typename Specialized = typename _Proc::specialized_types,
              std::enable_if_t<!(_Proc::EnablePostStepDoIt), int> = 0>
    static void DoIt(_Args&&...)
    {
        geantx::Log(kInfo) << GEANT_HERE << "post step disabled";
    }
};

//--------------------------------------------------------------------------------------//
}  // namespace mpl

template <typename ProcessType, typename ParticleType>
struct AtRest
{
    using mpl_type = mpl::AtRest<ProcessType, ParticleType>;

    template <typename _Track, typename _Tp, typename _Func>
    AtRest(size_t _N, _Track* _track, intmax_t* _doit_idx, _Tp* _doit_value,
           _Func* _doit_apply)
    {
        auto _value = mpl_type::GPIL(_track);
        if(_value < *_doit_value)
        {
            *_doit_idx   = _N;
            *_doit_value = _value;
            *_doit_apply = [&]() { mpl_type::DoIt(_track); };
        }
    }
};

template <typename ProcessType, typename ParticleType>
struct AlongStep
{
    using mpl_type = mpl::AlongStep<ProcessType, ParticleType>;

    template <typename _Track, typename _Tp, typename _Func>
    AlongStep(size_t _N, _Track*& _track, intmax_t* _doit_idx, _Tp* _doit_value,
              _Func* _doit_apply)
    {
        auto _value = mpl_type::GPIL(_track);
        if(_value < *_doit_value)
        {
            *_doit_idx   = _N;
            *_doit_value = _value;
            *_doit_apply = [&]() { mpl_type::DoIt(_track); };
        }
    }
};

template <typename ProcessType, typename ParticleType>
struct PostStep
{
    using mpl_type = mpl::PostStep<ProcessType, ParticleType>;

    template <typename _Track, typename _Tp, typename _Func>
    PostStep(size_t _N, _Track* _track, intmax_t* _doit_idx, _Tp* _doit_value,
             _Func* _doit_apply)
    {
        auto _value = mpl_type::GPIL(_track);
        if(_value < *_doit_value)
        {
            *_doit_idx   = _N;
            *_doit_value = _value;
            *_doit_apply = [=]() { mpl_type::DoIt(_track); };
        }
    }
};

}  // namespace geantx
