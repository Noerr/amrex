#ifndef AMREX_PARTICLEIO_H
#define AMREX_PARTICLEIO_H
#include <AMReX_Config.H>

#include <AMReX_ParticleHeader.H>
#include <AMReX_WriteBinaryParticleData.H>
#include <AMReX_VisMF.H>

#include <AMReX_Print.H>

#include <cstdlib>
#include <iterator>

namespace amrex {

template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WriteParticleRealData (void* data, size_t size, std::ostream& os) const
{
    if (sizeof(typename ParticleType::RealType) == 4) {
        writeFloatData((float*) data, size, os, ParticleRealDescriptor);
    }
    else if (sizeof(typename ParticleType::RealType) == 8) {
        writeDoubleData((double*) data, size, os, ParticleRealDescriptor);
    }
}

template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::ReadParticleRealData (void* data, size_t size, std::istream& is)
{
    if (sizeof(typename ParticleType::RealType) == 4) {
        readFloatData((float*) data, size, is, ParticleRealDescriptor);
    }
    else if (sizeof(typename ParticleType::RealType) == 8) {
        readDoubleData((double*) data, size, is, ParticleRealDescriptor);
    }
}

struct FilterPositiveID {
    template <typename P>
    AMREX_GPU_HOST_DEVICE
    int operator() (const P& p) const {
        return p.id().is_valid();
    }
};

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::Checkpoint (const std::string& dir,
              const std::string& name, bool /*is_checkpoint*/,
              const Vector<std::string>& real_comp_names,
              const Vector<std::string>& int_comp_names) const
{
    Vector<int> write_real_comp;
    Vector<std::string> tmp_real_comp_names;

    int first_rcomp = ParticleType::is_soa_particle ? AMREX_SPACEDIM : 0;
    for (int i = first_rcomp; i < NStructReal + NumRealComps(); ++i )
    {
        write_real_comp.push_back(1);
        if (real_comp_names.empty())
        {
            tmp_real_comp_names.push_back(getDefaultCompNameReal<ParticleType>(i));
        }
        else
        {
            tmp_real_comp_names.push_back(real_comp_names[i-first_rcomp]);
        }
    }

    Vector<int> write_int_comp;
    Vector<std::string> tmp_int_comp_names;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i )
    {
        write_int_comp.push_back(1);
        if (int_comp_names.empty())
        {
            tmp_int_comp_names.push_back(getDefaultCompNameInt<ParticleType>(i));
        }
        else
        {
            tmp_int_comp_names.push_back(int_comp_names[i]);
        }
    }

    WriteBinaryParticleData(dir, name, write_real_comp, write_int_comp,
                            tmp_real_comp_names, tmp_int_comp_names,
                            FilterPositiveID{}, true);
}

template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::Checkpoint (const std::string& dir, const std::string& name,
              const Vector<int>& write_real_comp,
              const Vector<int>& write_int_comp,
              const Vector<std::string>& real_comp_names,
              const Vector<std::string>& int_comp_names) const
{
    WriteBinaryParticleData(dir, name, write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names,
                            FilterPositiveID{}, true);
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFile (const std::string& dir, const std::string& name) const
{
    Vector<int> write_real_comp;
    Vector<std::string> real_comp_names;

    int first_rcomp = ParticleType::is_soa_particle ? AMREX_SPACEDIM : 0;
    for (int i = first_rcomp; i < NStructReal + NumRealComps(); ++i )
    {
        write_real_comp.push_back(1);
        real_comp_names.push_back(getDefaultCompNameReal<ParticleType>(i));
    }

    Vector<int> write_int_comp;
    Vector<std::string> int_comp_names;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i )
    {
        write_int_comp.push_back(1);
        int_comp_names.push_back(getDefaultCompNameInt<ParticleType>(i));
    }

    WriteBinaryParticleData(dir, name, write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names,
                            FilterPositiveID{});
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFile (const std::string& dir, const std::string& name,
                 const Vector<std::string>& real_comp_names,
                 const Vector<std::string>& int_comp_names) const
{
    if constexpr(ParticleType::is_soa_particle) {
        AMREX_ALWAYS_ASSERT(real_comp_names.size() == NumRealComps() + NStructReal - AMREX_SPACEDIM); // pure SoA: skip positions
    } else {
        AMREX_ALWAYS_ASSERT(real_comp_names.size() == NumRealComps() + NStructReal);
    }
    AMREX_ASSERT( int_comp_names.size() == NStructInt  + NumIntComps() );

    Vector<int> write_real_comp;
    int nrc = ParticleType::is_soa_particle ? NStructReal + NumRealComps() - AMREX_SPACEDIM : NStructReal + NumRealComps();
    for (int i = 0; i < nrc; ++i) {
        write_real_comp.push_back(1);
    }

    Vector<int> write_int_comp;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i) {
        write_int_comp.push_back(1);
    }

    WriteBinaryParticleData(dir, name,
                            write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names,
                            FilterPositiveID{});
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFile (const std::string& dir, const std::string& name,
                 const Vector<std::string>& real_comp_names) const
{
    if constexpr(ParticleType::is_soa_particle) {
        AMREX_ALWAYS_ASSERT(real_comp_names.size() == NumRealComps() + NStructReal - AMREX_SPACEDIM); // pure SoA: skip positions
    } else {
        AMREX_ALWAYS_ASSERT(real_comp_names.size() == NumRealComps() + NStructReal);
    }

    Vector<int> write_real_comp;
    int nrc = ParticleType::is_soa_particle ? NStructReal + NumRealComps() - AMREX_SPACEDIM : NStructReal + NumRealComps();
    for (int i = 0; i < nrc; ++i) {
        write_real_comp.push_back(1);
    }

    Vector<int> write_int_comp;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i) {
        write_int_comp.push_back(1);
    }

    Vector<std::string> int_comp_names;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i )
    {
        int_comp_names.push_back(getDefaultCompNameInt<ParticleType>(i));
    }

    WriteBinaryParticleData(dir, name,
                            write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names,
                            FilterPositiveID{});
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFile (const std::string& dir,
                 const std::string& name,
                 const Vector<int>& write_real_comp,
                 const Vector<int>& write_int_comp) const
{

    if constexpr(ParticleType::is_soa_particle) {
        AMREX_ALWAYS_ASSERT(write_real_comp.size() == NumRealComps() + NStructReal - AMREX_SPACEDIM); // pure SoA: skip positions
    } else {
        AMREX_ALWAYS_ASSERT(write_real_comp.size() == NumRealComps() + NStructReal);
    }
    AMREX_ASSERT(write_int_comp.size()  == NStructInt  + NumIntComps() );

    Vector<std::string> real_comp_names;
    int first_rcomp = ParticleType::is_soa_particle ? AMREX_SPACEDIM : 0;
    for (int i = first_rcomp; i < NStructReal + NumRealComps(); ++i )
    {
        real_comp_names.push_back(getDefaultCompNameReal<ParticleType>(i));
    }

    Vector<std::string> int_comp_names;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i )
    {
        int_comp_names.push_back(getDefaultCompNameInt<ParticleType>(i));
    }

    WriteBinaryParticleData(dir, name, write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names,
                            FilterPositiveID{});
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>::
WritePlotFile (const std::string& dir, const std::string& name,
               const Vector<int>& write_real_comp,
               const Vector<int>& write_int_comp,
               const Vector<std::string>& real_comp_names,
               const Vector<std::string>&  int_comp_names) const
{
    BL_PROFILE("ParticleContainer::WritePlotFile()");

    WriteBinaryParticleData(dir, name,
                            write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names,
                            FilterPositiveID{});
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
template <class F>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFile (const std::string& dir, const std::string& name, F const& f) const
{
    Vector<int> write_real_comp;
    Vector<std::string> real_comp_names;

    int first_rcomp = ParticleType::is_soa_particle ? AMREX_SPACEDIM : 0;
    for (int i = first_rcomp; i < NStructReal + NumRealComps(); ++i )
    {
        write_real_comp.push_back(1);
        real_comp_names.push_back(getDefaultCompNameReal<ParticleType>(i));
    }

    Vector<int> write_int_comp;
    Vector<std::string> int_comp_names;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i )
    {
        write_int_comp.push_back(1);
        int_comp_names.push_back(getDefaultCompNameInt<ParticleType>(i));
    }

    WriteBinaryParticleData(dir, name, write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names, f);
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
template <class F>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFile (const std::string& dir, const std::string& name,
                 const Vector<std::string>& real_comp_names,
                 const Vector<std::string>& int_comp_names, F const& f) const
{
    if constexpr(ParticleType::is_soa_particle) {
        AMREX_ALWAYS_ASSERT(real_comp_names.size() == NumRealComps() + NStructReal - AMREX_SPACEDIM); // pure SoA: skip positions
    } else {
        AMREX_ALWAYS_ASSERT(real_comp_names.size() == NumRealComps() + NStructReal);
    }
    AMREX_ASSERT( int_comp_names.size() == NStructInt  + NumIntComps() );

    Vector<int> write_real_comp;
    int nrc = ParticleType::is_soa_particle ? NStructReal + NumRealComps() - AMREX_SPACEDIM : NStructReal + NumRealComps();
    for (int i = 0; i < nrc; ++i) {
        write_real_comp.push_back(1);
    }

    Vector<int> write_int_comp;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i) {
        write_int_comp.push_back(1);
    }

    WriteBinaryParticleData(dir, name,
                            write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names, f);
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
template <class F>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFile (const std::string& dir, const std::string& name,
                 const Vector<std::string>& real_comp_names, F const& f) const
{
    if constexpr(ParticleType::is_soa_particle) {
        AMREX_ALWAYS_ASSERT(real_comp_names.size() == NumRealComps() + NStructReal - AMREX_SPACEDIM); // pure SoA: skip positions
    } else {
        AMREX_ALWAYS_ASSERT(real_comp_names.size() == NumRealComps() + NStructReal);
    }

    Vector<int> write_real_comp;
    int nrc = ParticleType::is_soa_particle ? NStructReal + NumRealComps() - AMREX_SPACEDIM : NStructReal + NumRealComps();
    for (int i = 0; i < nrc; ++i) {
        write_real_comp.push_back(1);
    }

    Vector<int> write_int_comp;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i) {
        write_int_comp.push_back(1);
    }

    Vector<std::string> int_comp_names;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i )
    {
        int_comp_names.push_back(getDefaultCompNameInt<ParticleType>(i));
    }

    WriteBinaryParticleData(dir, name,
                            write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names, f);
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
template <class F>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFile (const std::string& dir,
                 const std::string& name,
                 const Vector<int>& write_real_comp,
                 const Vector<int>& write_int_comp, F const& f) const
{
    if constexpr(ParticleType::is_soa_particle) {
        AMREX_ALWAYS_ASSERT(write_real_comp.size() == NumRealComps() + NStructReal - AMREX_SPACEDIM); // pure SoA: skip positions
    } else {
        AMREX_ALWAYS_ASSERT(write_real_comp.size() == NumRealComps() + NStructReal);
    }
    AMREX_ASSERT(write_int_comp.size()  == NStructInt  + NumIntComps() );

    Vector<std::string> real_comp_names;
    int first_rcomp = ParticleType::is_soa_particle ? AMREX_SPACEDIM : 0;
    for (int i = first_rcomp; i < NStructReal + NumRealComps(); ++i )
    {
        real_comp_names.push_back(getDefaultCompNameReal<ParticleType>(i));
    }

    Vector<std::string> int_comp_names;
    for (int i = 0; i < NStructInt + NumIntComps(); ++i )
    {
        int_comp_names.push_back(getDefaultCompNameInt<ParticleType>(i));
    }

    WriteBinaryParticleData(dir, name, write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names, f);
}

/// \ingroup amrex_io
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
template <class F>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>::
WritePlotFile (const std::string& dir, const std::string& name,
               const Vector<int>& write_real_comp,
               const Vector<int>& write_int_comp,
               const Vector<std::string>& real_comp_names,
               const Vector<std::string>&  int_comp_names,
               F const& f) const
{
    BL_PROFILE("ParticleContainer::WritePlotFile()");

    WriteBinaryParticleData(dir, name,
                            write_real_comp, write_int_comp,
                            real_comp_names, int_comp_names, f);
}

template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
template <class F>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WriteBinaryParticleData (const std::string& dir, const std::string& name,
                           const Vector<int>& write_real_comp,
                           const Vector<int>& write_int_comp,
                           const Vector<std::string>& real_comp_names,
                           const Vector<std::string>& int_comp_names,
                           F const& f, bool is_checkpoint) const
{
    if (AsyncOut::UseAsyncOut()) {
        WriteBinaryParticleDataAsync(*this, dir, name,
                                     write_real_comp, write_int_comp,
                                     real_comp_names, int_comp_names, is_checkpoint);
    } else
    {
        WriteBinaryParticleDataSync(*this, dir, name,
                                    write_real_comp, write_int_comp,
                                    real_comp_names, int_comp_names,
                                    f, is_checkpoint);
    }
}

template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::CheckpointPre ()
{
    if( ! usePrePost) {
        return;
    }

    BL_PROFILE("ParticleContainer::CheckpointPre()");

    const int IOProcNumber = ParallelDescriptor::IOProcessorNumber();
    Long nparticles = 0;
    Long  maxnextid  = ParticleType::NextID();

    for (int lev = 0; lev < std::ssize(m_particles);  lev++) {
        const auto& pmap = m_particles[lev];
        for (const auto& kv : pmap) {
            const auto& aos = kv.second.GetArrayOfStructs();
            for (int k = 0; k < aos.numParticles(); ++k) {
                const ParticleType& p = aos[k];
                if (p.id().is_valid()) {
                    //
                    // Only count (and checkpoint) valid particles.
                    //
                    nparticles++;
                }
            }
        }
    }
    ParallelDescriptor::ReduceLongSum(nparticles, IOProcNumber);

    ParticleType::NextID(maxnextid);
    ParallelDescriptor::ReduceLongMax(maxnextid, IOProcNumber);

    nparticlesPrePost = nparticles;
    maxnextidPrePost  = maxnextid;

    nParticlesAtLevelPrePost.clear();
    nParticlesAtLevelPrePost.resize(finestLevel() + 1, 0);
    for(int lev(0); lev <= finestLevel(); ++lev) {
        nParticlesAtLevelPrePost[lev] = NumberOfParticlesAtLevel(lev);
    }

    whichPrePost.clear();
    whichPrePost.resize(finestLevel() + 1);
    countPrePost.clear();
    countPrePost.resize(finestLevel() + 1);
    wherePrePost.clear();
    wherePrePost.resize(finestLevel() + 1);

    filePrefixPrePost.clear();
    filePrefixPrePost.resize(finestLevel() + 1);
}


template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::CheckpointPost ()
{
    if( ! usePrePost) {
        return;
    }

    BL_PROFILE("ParticleContainer::CheckpointPost()");

    const int IOProcNumber = ParallelDescriptor::IOProcessorNumber();
    std::ofstream HdrFile;
    HdrFile.open(HdrFileNamePrePost.c_str(), std::ios::out | std::ios::app);

    for(int lev(0); lev <= finestLevel(); ++lev) {
        ParallelDescriptor::ReduceIntSum (whichPrePost[lev].dataPtr(), whichPrePost[lev].size(), IOProcNumber);
        ParallelDescriptor::ReduceIntSum (countPrePost[lev].dataPtr(), countPrePost[lev].size(), IOProcNumber);
        ParallelDescriptor::ReduceLongSum(wherePrePost[lev].dataPtr(), wherePrePost[lev].size(), IOProcNumber);


        if(ParallelDescriptor::IOProcessor()) {
            for(int j(0); j < whichPrePost[lev].size(); ++j) {
                HdrFile << whichPrePost[lev][j] << ' ' << countPrePost[lev][j] << ' ' << wherePrePost[lev][j] << '\n';
            }

            const bool gotsome = (nParticlesAtLevelPrePost[lev] > 0);
            if(gotsome && doUnlink) {
//            BL_PROFILE_VAR("PC<NNNN>::Checkpoint:unlink", unlink_post);
                // Unlink any zero-length data files.
                Vector<Long> cnt(nOutFilesPrePost,0);

                for(int i(0), N = countPrePost[lev].size(); i < N; ++i) {
                    cnt[whichPrePost[lev][i]] += countPrePost[lev][i];
                }

                for(int i = 0; i < std::ssize(cnt); ++i) {
                    if(cnt[i] == 0) {
                        std::string FullFileName = NFilesIter::FileName(i, filePrefixPrePost[lev]);
                        FileSystem::Remove(FullFileName);
                    }
                }
            }
        }
    }

    if(ParallelDescriptor::IOProcessor()) {
        HdrFile.flush();
        HdrFile.close();
        if( ! HdrFile.good()) {
            amrex::Abort("ParticleContainer::CheckpointPost(): problem writing HdrFile");
        }
    }
}

template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFilePre ()
{
    CheckpointPre();
}


template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WritePlotFilePost ()
{
    CheckpointPost();
}


template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WriteParticles (int lev, std::ofstream& ofs, int fnum,
                  Vector<int>& which, Vector<int>& count, Vector<Long>& where,
                  const Vector<int>& write_real_comp,
                  const Vector<int>& write_int_comp,
                  const Vector<std::map<std::pair<int, int>, FlagsVector>>& particle_io_flags,
                  bool is_checkpoint) const
{
    BL_PROFILE("ParticleContainer::WriteParticles()");

    // For a each grid, the tiles it contains
    std::map<int, Vector<int> > tile_map;

    for (const auto& kv : m_particles[lev])
    {
        const int grid = kv.first.first;
        const int tile = kv.first.second;
        tile_map[grid].push_back(tile);
        const auto& pflags = particle_io_flags[lev].at(kv.first);

        // Only write out valid particles.
        count[grid] += particle_detail::countFlags(pflags);
    }

    MFInfo info;
    info.SetAlloc(false);
    MultiFab state(ParticleBoxArray(lev), ParticleDistributionMap(lev), 1,0,info);

    for (MFIter mfi(state); mfi.isValid(); ++mfi)
    {
        const int grid = mfi.index();

        which[grid] = fnum;
        where[grid] = VisMF::FileOffset(ofs);

        if (count[grid] <= 0) { continue; }

        Vector<int> istuff;
        Vector<ParticleReal> rstuff;
        particle_detail::packIOData(istuff, rstuff, *this, lev, grid,
                                    write_real_comp, write_int_comp,
                                    particle_io_flags, tile_map[grid], count[grid], is_checkpoint);

        writeIntData(istuff.dataPtr(), istuff.size(), ofs);
        if (!VisMF::GetNoFlushAfterWrite()) {
            ofs.flush();  // Some systems require this flush() (probably due to a bug)
        }

        WriteParticleRealData(rstuff.dataPtr(), rstuff.size(), ofs);
        if (!VisMF::GetNoFlushAfterWrite()) {
            ofs.flush();  // Some systems require this flush() (probably due to a bug)
        }
    }
}


template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::Restart (const std::string& dir, const std::string& file, bool /*is_checkpoint*/)
{
    Restart(dir, file);
}

template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::Restart (const std::string& dir, const std::string& file)
{
    BL_PROFILE("ParticleContainer::Restart()");
    AMREX_ASSERT(!dir.empty());
    AMREX_ASSERT(!file.empty());

    const auto strttime = amrex::second();

    int DATA_Digits_Read(5);
    ParmParse pp("particles");
    pp.query("datadigits_read",DATA_Digits_Read);

    std::string fullname = dir;
    if (!fullname.empty() && fullname[fullname.size()-1] != '/') {
        fullname += '/';
    }
    fullname += file;
    std::string HdrFileName = fullname;
    if (!HdrFileName.empty() && HdrFileName[HdrFileName.size()-1] != '/') {
        HdrFileName += '/';
    }
    HdrFileName += "Header";

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(HdrFileName, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream HdrFile(fileCharPtrString, std::istringstream::in);

    // The metadata prefix (version through finest level) and the per-level
    // grid table (data file index, particle count and byte offset per grid)
    // are parsed by the shared ParticleHeader.
    // See AMReX_ParticleHeader.{H,cpp}.
    ParticleHeader header;
    header.parse(HdrFile);
    header.parse_grid_table(HdrFile);

    const std::string& how = header.how;
    const bool convert_ids = header.convert_ids;

    if (header.dim != AMREX_SPACEDIM) {
        amrex::Abort("ParticleContainer::Restart(): dm != AMREX_SPACEDIM");
    }

    const int nrc = ParticleType::is_soa_particle ? NStructReal + NumRealComps() - AMREX_SPACEDIM : NStructReal + NumRealComps();
    if (header.num_real != nrc) {
        amrex::Abort("ParticleContainer::Restart(): nr not the expected value");
    }

    if (header.num_int != NStructInt + NumIntComps()) {
        amrex::Abort("ParticleContainer::Restart(): ni != NStructInt");
    }

    AMREX_ASSERT(header.num_particles >= 0);

    AMREX_ASSERT(header.next_id > 0);
    ParticleType::NextID(header.next_id);

    const int finest_level_in_file = header.finest_level;
    AMREX_ASSERT(finest_level_in_file >= 0);

    // Read the BoxArrays stored in the checkpoint for each level so we can
    // drive the read using the file's layout.  The container's own grids are
    // never modified: Redistribute() will move every particle to the correct
    // tile on whatever grids the container currently uses.
    Vector<BoxArray>            file_bas(finest_level_in_file + 1);
    Vector<DistributionMapping> file_dms(finest_level_in_file + 1);
    for (int lev = 0; lev <= finest_level_in_file; lev++)
    {
        std::string phdr_name = fullname;
        phdr_name += "/Level_";
        phdr_name = amrex::Concatenate(phdr_name, lev, 1);
        phdr_name += "/Particle_H";

        if (amrex::FileExists(phdr_name)) {
            Vector<char> phdr_chars;
            ParallelDescriptor::ReadAndBcastFile(phdr_name, phdr_chars);
            std::string phdr_string(phdr_chars.dataPtr());
            std::istringstream phdr_file(phdr_string, std::istringstream::in);
            file_bas[lev].readFrom(phdr_file);
        } else {
            // No particles were written at this level; use a single-box BA as
            // a placeholder so that MFIter below works uniformly.
            file_bas[lev] = BoxArray(Geom(lev).Domain());
        }
        // If the checkpoint's box array is the one this container is already
        // using, read each box on the rank that owns it. Particles then land
        // directly in their final tile, and the Redistribute() at the end of
        // Restart() has nothing to move: it takes the
        // tot_snds == 0 && tot_rcvs == 0 early-out and allocates no send or
        // receive buffer at all.
        //
        // This is the pre-#5497 behaviour for the common case, restored here
        // because the memory cost of not doing it is severe at scale. Reading
        // a box that belongs to another rank means holding, concurrently: the
        // foreign box's tile, a send buffer sized to essentially all of it, a
        // receive buffer for this rank's own box arriving from elsewhere, and
        // the destination tile -- several times the steady-state particle
        // payload. That caps how much device memory a run can use in steady
        // state and still be restartable, which directly limits strong
        // scaling.
        //
        // #5497's purpose -- not requiring the caller's grids to match the
        // file's -- is unaffected: when the box arrays genuinely differ, the
        // original path below is taken unchanged.
        const bool ba_matches =
            (lev <= finestLevel()) && (file_bas[lev] == ParticleBoxArray(lev));
        if (std::getenv("AMREX_RESTART_DEBUG") != nullptr) {
            amrex::Print()
                << "[restart-dbg] lev=" << lev
                << " boxarray_matches=" << (ba_matches ? "yes" : "no")
                << " -> read_dm=" << (ba_matches ? "container (no shuffle)"
                                                 : "file-derived (redistribute)")
                << "\n";
        }
        if (ba_matches) {
            file_dms[lev] = ParticleDistributionMap(lev);
        } else {
            const int NReaders = MaxReaders();
            if (ParallelDescriptor::NProcs() <= NReaders) {
                file_dms[lev] = DistributionMapping(file_bas[lev]);
            } else {
                Vector<int> pmap(file_bas[lev].size());
                for (int i = 0; i < (int)pmap.size(); ++i) { pmap[i] = i % NReaders; }
                file_dms[lev] = DistributionMapping(pmap);
            }
        }
    }

    resizeData();

    if (finest_level_in_file > finestLevel()) {
        m_particles.resize(finest_level_in_file+1);
    }

    for (int lev = 0; lev <= finest_level_in_file; lev++) {
        auto const& grid_table = header.grids[lev];

        // Use the file's own layout to distribute reading across ranks.
        MFInfo info;
        info.SetAlloc(false);
        MultiFab file_mf(file_bas[lev], file_dms[lev], 1, 0, info);
        Vector<int> grids_to_read;
        for (MFIter mfi(file_mf); mfi.isValid(); ++mfi) {
            grids_to_read.push_back(mfi.index());
        }

        // TEMPORARY DIAGNOSTIC / EXPERIMENT SCAFFOLDING -- not for upstream.
        //
        //   AMREX_RESTART_DEBUG=1    log the read layout and every dst_tile
        //                            resize, to establish how many boxes this
        //                            rank reads, how big they are, and whether
        //                            any destination tile is grown more than
        //                            once (which would mean realloc churn).
        //   AMREX_RESTART_PRESIZE=1  reserve each destination tile to its known
        //                            final size before reading, so that the
        //                            resize inside ReadParticles cannot
        //                            reallocate. A no-op if each tile is only
        //                            ever sized once.
        //
        static const bool restart_dbg =
            (std::getenv("AMREX_RESTART_DEBUG") != nullptr);
        static const bool restart_presize =
            (std::getenv("AMREX_RESTART_PRESIZE") != nullptr);

        if (restart_dbg) {
            Long total_here = 0;
            Long max_here = 0;
            for (int grid : grids_to_read) {
                total_here += grid_table[grid].count;
                max_here = std::max(max_here, Long(grid_table[grid].count));
            }
            amrex::AllPrint()
                << "[restart-dbg] rank=" << ParallelDescriptor::MyProc()
                << " lev=" << lev
                << " boxes_in_file=" << grid_table.size()
                << " boxes_this_rank=" << grids_to_read.size()
                << " particles_this_rank=" << total_here
                << " largest_box=" << max_here
                << "\n";
            for (int grid : grids_to_read) {
                amrex::AllPrint()
                    << "[restart-dbg]   rank=" << ParallelDescriptor::MyProc()
                    << " lev=" << lev << " grid=" << grid
                    << " count=" << grid_table[grid].count
                    << " file=" << grid_table[grid].which
                    << "\n";
            }
        }

        if (restart_presize) {
            // Destination tile for a given file box is keyed by that box's
            // grid index, so the final size is known from the header before
            // any particle is read.
            for (int grid : grids_to_read) {
                if (grid_table[grid].count <= 0) { continue; }
                auto& dst_tile = DefineAndReturnParticleTile(lev, grid, 0);
                const auto want = dst_tile.size() + std::size_t(grid_table[grid].count);
                if (restart_dbg) {
                    amrex::AllPrint()
                        << "[restart-presize] rank=" << ParallelDescriptor::MyProc()
                        << " lev=" << lev << " grid=" << grid
                        << " reserve=" << want << "\n";
                }
                // Exact, not the default Poisson: the whole point is to
                // allocate the known final size once, with no headroom.
                dst_tile.reserve(want, GrowthStrategy::Exact);
            }
        }

        for(int grid : grids_to_read) {
            if (grid_table[grid].count <= 0) { continue; }

            // The file names in the header file are relative.
            std::string name = fullname;

            if (!name.empty() && name[name.size()-1] != '/') {
                name += '/';
            }

            name += "Level_";
            name += amrex::Concatenate("", lev, 1);
            name += '/';
            name += DataPrefix();
            name += amrex::Concatenate("", grid_table[grid].which, DATA_Digits_Read);

            std::ifstream ParticleFile;

            ParticleFile.open(name.c_str(), std::ios::in | std::ios::binary);

            if (!ParticleFile.good()) {
                amrex::FileOpenFailed(name);
            }

            ParticleFile.seekg(grid_table[grid].where, std::ios::beg);

            // Use if constexpr to avoid instantiating the mis-matched
            // type case and triggering the static_assert on the
            // underlying copy calls
            if (how == "single") {
                if constexpr (std::is_same_v<ParticleReal, float>) {
                    ReadParticles<float>(grid_table[grid].count, grid, lev, ParticleFile, finest_level_in_file, convert_ids);
                } else {
                    amrex::Error("File contains single-precision data, while AMReX is compiled with ParticleReal==double");
                }
            }
            else if (how == "double") {
                if constexpr (std::is_same_v<ParticleReal, double>) {
                    ReadParticles<double>(grid_table[grid].count, grid, lev, ParticleFile, finest_level_in_file, convert_ids);
                } else {
                    amrex::Error("File contains double-precision data, while AMReX is compiled with ParticleReal==float");
                }
            }
            else {
                std::string msg("ParticleContainer::Restart(): bad parameter: ");
                msg += how;
                amrex::Error(msg.c_str());
            }

            ParticleFile.close();

            if (!ParticleFile.good()) {
                amrex::Abort("ParticleContainer::Restart(): problem reading particles");
            }
        }
    }

    Redistribute();

    AMREX_ASSERT(OK());

    if (m_verbose > 1) {
        auto stoptime = amrex::second() - strttime;
        ParallelDescriptor::ReduceRealMax(stoptime, ParallelDescriptor::IOProcessorNumber());
        amrex::Print() << "ParticleContainer::Restart() time: " << stoptime << '\n';
    }
}

// Read a batch of particles from the checkpoint file
template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
template <class RTYPE>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::ReadParticles (int cnt, int grd, int lev, std::ifstream& ifs,
                 int finest_level_in_file, bool convert_ids)
{
    BL_PROFILE("ParticleContainer::ReadParticles()");
    AMREX_ASSERT(cnt > 0);
    AMREX_ASSERT(lev < std::ssize(m_particles));

    // First read in the integer data in binary.  We do not store
    // the m_lev and m_grid data on disk.  We can easily recreate
    // that given the structure of the checkpoint file.
    const int iChunkSize = 2 + NStructInt + NumIntComps();
    Vector<int> istuff(std::size_t(cnt)*iChunkSize);
    readIntData(istuff.dataPtr(), istuff.size(), ifs, FPC::NativeIntDescriptor());

    // Then the real data in binary.  Instead of reading this grid's entire real
    // block into one full-box buffer (which would coexist with the reassembled
    // host_* buffers below, holding two full-box host copies at once), stream it
    // in bounded sequential segments.  istuff already holds the whole int block,
    // and this grid's real block is contiguous in the file, so reading it in
    // chunks of at most chunk_np particles caps the host transient at ~1x the
    // box instead of ~2x -- matching the checkpoint write footprint and letting
    // large-particle checkpoints restart within host DRAM.
    const int rChunkSize = ParticleType::is_soa_particle ? NStructReal + NumRealComps() : AMREX_SPACEDIM + NStructReal + NumRealComps();

    // Target ~256 MiB of real data per read segment; clamp to [1, cnt].
    constexpr std::size_t target_seg_bytes = std::size_t(256) << 20;
    const std::size_t seg_bytes_per_part = std::max(std::size_t(1),
        std::size_t(rChunkSize) * sizeof(RTYPE));
    const std::size_t chunk_np = std::max(std::size_t(1),
        std::min(std::size_t(cnt), target_seg_bytes / seg_bytes_per_part));

    // Now reassemble the particles.  iptr walks the resident int block; rptr is
    // (re)pointed at each freshly read real-data segment inside the loop below.
    Vector<RTYPE> rstuff;
    int*   iptr = istuff.dataPtr();
    RTYPE* rptr = nullptr;
    std::size_t seg_end = 0;

    Particle<NStructReal, NStructInt> ptemp;
    ParticleLocData pld;

    Vector<std::map<std::pair<int, int>, Gpu::HostVector<ParticleType> > > host_particles;
    host_particles.reserve(15);
    host_particles.resize(finest_level_in_file+1);

    Vector<std::map<std::pair<int, int>,
                    std::vector<Gpu::HostVector<RTYPE> > > > host_real_attribs;
    host_real_attribs.reserve(15);
    host_real_attribs.resize(finest_level_in_file+1);

    Vector<std::map<std::pair<int, int>,
                    std::vector<Gpu::HostVector<int> > > > host_int_attribs;
    host_int_attribs.reserve(15);
    host_int_attribs.resize(finest_level_in_file+1);

    Vector<std::map<std::pair<int, int>, Gpu::HostVector<uint64_t> > > host_idcpu;
    host_idcpu.reserve(15);
    host_idcpu.resize(finest_level_in_file+1);

    for (int i = 0; i < cnt; i++) {
        // Load the next contiguous real-data segment when the current one is
        // exhausted.  rptr is reset to the new segment; iptr is not (the whole
        // int block is already resident and walked continuously).
        if (std::size_t(i) >= seg_end) {
            const std::size_t seg_np = std::min(chunk_np, std::size_t(cnt - i));
            rstuff.resize(seg_np * std::size_t(rChunkSize));
            ReadParticleRealData(rstuff.dataPtr(), rstuff.size(), ifs);
            rptr = rstuff.dataPtr();
            seg_end += seg_np;
        }

        // note: for pure SoA particle layouts, we do write the id, cpu and positions as a struct
        //       for backwards compatibility with readers
        if (convert_ids) {
            std::int32_t  xi, yi;
            std::uint32_t xu, yu;
            xi = iptr[0];
            yi = iptr[1];
            std::memcpy(&xu, &xi, sizeof(xi));
            std::memcpy(&yu, &yi, sizeof(yi));
            ptemp.m_idcpu = ((std::uint64_t)xu) << 32 | yu;
        } else {
            ptemp.id()   = iptr[0];
            ptemp.cpu()  = iptr[1];
        }
        iptr += 2;

        for (int j = 0; j < NStructInt; j++)
        {
            ptemp.idata(j) = *iptr;
            ++iptr;
        }

        AMREX_ASSERT(ptemp.id().is_valid());

        AMREX_D_TERM(ptemp.pos(0) = ParticleReal(rptr[0]);,
                     ptemp.pos(1) = ParticleReal(rptr[1]);,
                     ptemp.pos(2) = ParticleReal(rptr[2]););

        rptr += AMREX_SPACEDIM;

        for (int j = 0; j < NStructReal; j++)
        {
            ptemp.rdata(j) = ParticleReal(*rptr);
            ++rptr;
        }

        locateParticle(ptemp, pld, 0, finestLevel(), IntVect(0));

        std::pair<int, int> ind(grd, pld.m_tile);

        host_real_attribs[lev][ind].resize(NumRealComps());
        host_int_attribs[lev][ind].resize(NumIntComps());

        // add the struct
        if constexpr(!ParticleType::is_soa_particle)
        {
            host_particles[lev][ind].push_back(ptemp);

            // add the real...
            for (int icomp = 0; icomp < NumRealComps(); icomp++) {
                host_real_attribs[lev][ind][icomp].push_back(*rptr);
                ++rptr;
            }

            // ... and int array data
            for (int icomp = 0; icomp < NumIntComps(); icomp++) {
                host_int_attribs[lev][ind][icomp].push_back(*iptr);
                ++iptr;
            }
        } else {
            host_particles[lev][ind];

            for (int j = 0; j < AMREX_SPACEDIM; j++) {
                host_real_attribs[lev][ind][j].push_back(ptemp.pos(j));
            }

            host_idcpu[lev][ind].push_back(ptemp.m_idcpu);

            // read all other SoA
            // add the real...
            for (int icomp = AMREX_SPACEDIM; icomp < NumRealComps(); icomp++) {
                host_real_attribs[lev][ind][icomp].push_back(*rptr);
                ++rptr;
            }

            // ... and int array data
            for (int icomp = 0; icomp < NumIntComps(); icomp++) {
                host_int_attribs[lev][ind][icomp].push_back(*iptr);
                ++iptr;
            }
        }
    }

    for (int host_lev = 0; host_lev < std::ssize(host_particles); ++host_lev)
    {
        for (auto& kv : host_particles[host_lev]) {
            auto grid = kv.first.first;
            auto tile = kv.first.second;
            const auto& src_tile = kv.second;

            auto& dst_tile = DefineAndReturnParticleTile(host_lev, grid, tile);
            auto old_size = dst_tile.size();
            auto new_size = old_size;
            if constexpr(!ParticleType::is_soa_particle)
            {
                new_size += src_tile.size();
            } else {
                amrex::ignore_unused(src_tile);
                new_size += host_real_attribs[host_lev][std::make_pair(grid,tile)][0].size();
            }
            // TEMPORARY DIAGNOSTIC -- not for upstream. See AMREX_RESTART_DEBUG
            // above. old==0 on every line means each destination tile is sized
            // exactly once and there is no realloc churn to remove; old>0 means
            // the tile is being grown and AMREX_RESTART_PRESIZE should help.
            {
                static const bool restart_dbg_r =
                    (std::getenv("AMREX_RESTART_DEBUG") != nullptr);
                if (restart_dbg_r) {
                    const std::size_t psize_bytes =
                        sizeof(uint64_t)
                        + std::size_t(NumRealComps())*sizeof(ParticleReal)
                        + std::size_t(NumIntComps())*sizeof(int);
                    amrex::AllPrint()
                        << "[restart-dbg] resize rank=" << ParallelDescriptor::MyProc()
                        << " lev=" << host_lev
                        << " grid=" << grid << " tile=" << tile
                        << " old=" << old_size << " new=" << new_size
                        << " cap=" << dst_tile.capacity()
                        << " delta_MiB="
                        << ((new_size - old_size)*psize_bytes)/(1024*1024)
                        << "\n";
                }
            }

            dst_tile.resize(new_size);

            if constexpr(!ParticleType::is_soa_particle)
            {
                Gpu::copyAsync(Gpu::hostToDevice, src_tile.begin(), src_tile.end(),
                               dst_tile.GetArrayOfStructs().begin() + old_size);
            } else {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_idcpu[host_lev][std::make_pair(grid,tile)].begin(),
                               host_idcpu[host_lev][std::make_pair(grid,tile)].end(),
                               dst_tile.GetStructOfArrays().GetIdCPUData().begin() + old_size);
            }

            for (int i = 0; i < NumRealComps(); ++i) { // NOLINT(readability-misleading-indentation)
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_real_attribs[host_lev][std::make_pair(grid,tile)][i].begin(),
                               host_real_attribs[host_lev][std::make_pair(grid,tile)][i].end(),
                               dst_tile.GetStructOfArrays().GetRealData(i).begin() + old_size);
            }

            for (int i = 0; i < NumIntComps(); ++i) {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_int_attribs[host_lev][std::make_pair(grid,tile)][i].begin(),
                               host_int_attribs[host_lev][std::make_pair(grid,tile)][i].end(),
                               dst_tile.GetStructOfArrays().GetIntData(i).begin() + old_size);
            }
        }
    }

    Gpu::streamSynchronize();
}

template <typename ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator, class CellAssignor>
void
ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>
::WriteAsciiFile (const std::string& filename)
{
    BL_PROFILE("ParticleContainer::WriteAsciiFile()");
    AMREX_ASSERT(!filename.empty());

    const auto strttime = amrex::second();
    //
    // Count # of valid particles.
    //
    Long nparticles = 0;

    for (int lev = 0; lev < std::ssize(m_particles);  lev++) {
        auto& pmap = m_particles[lev];
        for (const auto& kv : pmap) {
            const auto& aos = kv.second.GetArrayOfStructs();
            auto np = aos.numParticles();
            Gpu::HostVector<ParticleType> host_aos(np);
            Gpu::copy(Gpu::deviceToHost, aos.begin(), aos.begin() + np, host_aos.begin());
            for (int k = 0; k < np; ++k) {
                const ParticleType& p = host_aos[k];
                if (p.id().is_valid()) {
                    //
                    // Only count (and checkpoint) valid particles.
                    //
                    nparticles++;
                }
            }
        }
    }

    //
    // And send count to I/O processor.
    //
    ParallelDescriptor::ReduceLongSum(nparticles,ParallelDescriptor::IOProcessorNumber());

    if (ParallelDescriptor::IOProcessor())
    {
        //
        // Have I/O processor open file and write out particle metadata.
        //
        std::ofstream File;

        File.open(filename.c_str(), std::ios::out|std::ios::trunc);

        if (!File.good()) {
            amrex::FileOpenFailed(filename);
        }

        File << nparticles  << '\n';
        File << NStructReal << '\n';
        File << NStructInt  << '\n';
        File << NumRealComps()  << '\n';
        File << NumIntComps()   << '\n';

        File.flush();

        File.close();

        if (!File.good()) {
            amrex::Abort("ParticleContainer::WriteAsciiFile(): problem writing file");
        }
    }

    ParallelDescriptor::Barrier();

    const int MyProc = ParallelDescriptor::MyProc();

    for (int proc = 0; proc < ParallelDescriptor::NProcs(); proc++)
    {
        if (MyProc == proc)
        {
            //
            // Each CPU opens the file for appending and adds its particles.
            //
            VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

            std::ofstream File;

            File.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

            File.open(filename.c_str(), std::ios::out|std::ios::app);

            File.precision(15);

            if (!File.good()) {
                amrex::FileOpenFailed(filename);
            }

            for (int lev = 0; lev < std::ssize(m_particles);  lev++) {
                auto& pmap = m_particles[lev];
                for (const auto& kv : pmap) {
                    ParticleTile<ParticleType, NArrayReal, NArrayInt,
                                 amrex::PolymorphicArenaAllocator> pinned_ptile;
                    pinned_ptile.define(NumRuntimeRealComps(), NumRuntimeIntComps(),
                                        nullptr, nullptr, The_Pinned_Arena());
                    pinned_ptile.resize(kv.second.numParticles());
                    amrex::copyParticles(pinned_ptile, kv.second);
                    const auto& host_aos = pinned_ptile.GetArrayOfStructs();
                    const auto& host_soa = pinned_ptile.GetStructOfArrays();

                    auto np = host_aos.numParticles();
                    for (int index = 0; index < np; ++index) {
                        const ParticleType* it = &host_aos[index];
                        if (it->id().is_valid()) {

                            // write out the particle struct first...
                            AMREX_D_TERM(File << it->pos(0) << ' ',
                                              << it->pos(1) << ' ',
                                              << it->pos(2) << ' ');

                            for (int i = 0; i < NStructReal; i++) {
                                File << it->rdata(i) << ' ';
                            }

                            File << it->id()  << ' ';
                            File << it->cpu() << ' ';

                            for (int i = 0; i < NStructInt; i++) {
                                File << it->idata(i) << ' ';
                            }

                            // then the particle attributes.
                            for (int i = 0; i < NumRealComps(); i++) {
                                File << host_soa.GetRealData(i)[index] << ' ';
                            }

                            for (int i = 0; i < NumIntComps(); i++) {
                                File << host_soa.GetIntData(i)[index] << ' ';
                            }

                            File << '\n';
                        }
                    }
                }
            }

            File.flush();

            File.close();

            if (!File.good()) {
                amrex::Abort("ParticleContainer::WriteAsciiFile(): problem writing file");
            }

        }

        ParallelDescriptor::Barrier();
    }

    if (m_verbose > 1)
    {
        auto stoptime = amrex::second() - strttime;

        ParallelDescriptor::ReduceRealMax(stoptime,ParallelDescriptor::IOProcessorNumber());

        amrex::Print() << "ParticleContainer::WriteAsciiFile() time: " << stoptime << '\n';
    }
}

}

#endif /*AMREX_PARTICLEIO_H*/
