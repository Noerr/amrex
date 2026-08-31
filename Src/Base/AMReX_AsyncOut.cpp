#include <AMReX_AsyncOut.H>
#include <AMReX_BackgroundThread.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <atomic>
#include <ctime>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX.H>

namespace amrex::AsyncOut {

namespace {

bool s_asyncout = false;
int s_noutfiles = 64;
MPI_Comm s_comm = MPI_COMM_NULL;

std::unique_ptr<BackgroundThread> s_thread;

WriteInfo s_info;

}

void Initialize ()
{
    amrex::ignore_unused(s_comm,s_info);

    ParmParse pp("amrex");
    pp.queryAdd("async_out", s_asyncout);
    pp.queryAdd("async_out_nfiles", s_noutfiles);

    int nprocs = ParallelDescriptor::NProcs();
    s_noutfiles = std::min(s_noutfiles, nprocs);

#ifdef AMREX_USE_MPI
    if (s_asyncout && s_noutfiles < nprocs)
    {
        int provided = -1;
        MPI_Query_thread(&provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            amrex::Abort("AsyncOut with " + std::to_string(s_noutfiles) + " and "
                         + std::to_string(nprocs) + " processes requires "
                         + "MPI_THREAD_MULTIPLE at runtime, but got "
                         + ParallelDescriptor::mpi_level_to_string(provided));
        }
        int myproc = ParallelDescriptor::MyProc();
        s_info = GetWriteInfo(myproc);
        MPI_Comm_split(ParallelDescriptor::Communicator(), s_info.ifile, myproc, &s_comm);
    }
#endif

    if (s_asyncout) {
        s_thread = std::make_unique<BackgroundThread>();
    }

    ExecOnFinalize(Finalize);
}

void Finalize ()
{
    // Destroying the BackgroundThread submits a sentinel job and then joins,
    // so this blocks until every queued asynchronous write has completed. That
    // drain is real wall time at the end of a run and was previously invisible
    // to the profiler, which finalized before the finalize-function stack ran.
    // The write itself happens on the background thread and cannot be profiled
    // here -- TinyProfiler keeps a single region stack and is not thread-aware
    // -- but the main thread's wait for it is exactly the quantity of interest.
    BL_PROFILE("AsyncOut::Finalize()");

    if (s_thread) {
        // Announce before blocking, and report after. amrex::Print flushes on
        // both sides of its write, so the first line reaches the log before the
        // join begins. That matters for two reasons:
        //
        //   - a run that appears hung at exit can be identified as waiting on
        //     output rather than stuck elsewhere, without attaching a debugger;
        //   - the second line is positive evidence that asynchronous writes
        //     actually completed. File count, size and mtime do not establish
        //     that a checkpoint was fully written -- only that the writer got
        //     far enough to create the entries. This is the "true completion"
        //     signal that inspecting the output directory cannot provide.
        //
        // The profiler entry for this region gives the same duration, but only
        // if the run reaches the end and prints its report; these lines survive
        // a job that is killed mid-drain, which is exactly the case of interest.
        // Qualified by rank for the same reason as the per-job line above: this
        // is the reporting rank's own writer thread draining, not a statement
        // about every rank.
        amrex::Print() << "AsyncOut::Finalize(): rank " << ParallelDescriptor::MyProc()
                       << " waiting for asynchronous output to drain\n";
        const double t_start = amrex::second();

        s_thread.reset();

        amrex::Print() << "AsyncOut::Finalize(): rank " << ParallelDescriptor::MyProc()
                       << " drained in " << amrex::second() - t_start << " s\n";
    }

#ifdef AMREX_USE_MPI
    if (s_comm != MPI_COMM_NULL) { MPI_Comm_free(&s_comm); }
    s_comm = MPI_COMM_NULL;
#endif
}

bool UseAsyncOut () { return s_asyncout; }

WriteInfo GetWriteInfo (int rank)
{
    const int nfiles = s_noutfiles;
    const int nprocs = ParallelDescriptor::NProcs();
    const int nmaxspots = (nprocs + (nfiles-1)) / nfiles;  // max spots per file
    const int nfull = nfiles + nprocs - nmaxspots*nfiles;  // the first nfull files are full

    int ifile, ispot, nspots;
    if (rank < nfull*nmaxspots) {
        ifile = rank / nmaxspots;
        ispot = rank - ifile*nmaxspots;
        nspots = nmaxspots;
    } else {
        int tmpproc = rank-nfull*nmaxspots;
        ifile = tmpproc/(nmaxspots-1);
        ispot = tmpproc - ifile*(nmaxspots-1);
        ifile += nfull;
        nspots = nmaxspots - 1;
    }

    return WriteInfo{.ifile = ifile, .ispot = ispot, .nspots = nspots};
}

namespace {

std::atomic<int> s_job_seq{0};

// Wrap a submitted job so that its completion is reported from the background
// thread, at the moment it finishes, without the main thread waiting for
// anything. Asynchronous writes can run for many minutes while the simulation
// continues, and until now nothing recorded when one actually finished: the
// only guaranteed drain is at program exit, which for a run checkpointing every
// few hours may be a very long time after the fact, or may never happen if the
// job is killed.
//
// The line carries a wall-clock timestamp so it can be correlated with the
// simulation's own output, and the elapsed write time. Checkpoints separated by
// hours make the timestamp unambiguous on its own; a label would be needed only
// if submissions became closely spaced.
std::function<void()> with_completion_log (std::function<void()> f)
{
    const int seq = s_job_seq++;
    return [f = std::move(f), seq] () {
        const double t_start = amrex::second();

        f();

        const double dt = amrex::second() - t_start;

        std::time_t const t_now = std::time(nullptr);
        std::tm tm_buf{};
        char stamp[32] = "unknown";
#if defined(_WIN32)
        if (localtime_s(&tm_buf, &t_now) == 0) {
            std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm_buf);
        }
#else
        // localtime_r rather than localtime: this runs on the background
        // thread, and localtime returns a pointer to shared static storage.
        if (localtime_r(&t_now, &tm_buf) != nullptr) {
            std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm_buf);
        }
#endif

        // Built as a single line and written with one insertion, to minimise
        // interleaving with main-thread output.
        //
        // Named as one rank's completion, deliberately. amrex::Print reports
        // from the I/O process only, so this says nothing about the other
        // ranks: with fewer files than ranks, the ranks sharing a file write in
        // turn via the Wait/Notify baton, and the reporting rank is generally
        // early in its group. Treat it as a lower bound on when the output as a
        // whole was complete, not as a statement that it was.
        amrex::Print() << "AsyncOut: rank " << ParallelDescriptor::MyProc()
                       << " finished job " << seq << " at " << stamp
                       << " after " << dt << " s\n";
    };
}

}

void Submit (std::function<void()>&& a_f)
{
    s_thread->Submit(with_completion_log(std::move(a_f)));
}

void Submit (std::function<void()> const& a_f)
{
    s_thread->Submit(with_completion_log(a_f));
}

void Finish ()
{
    if (s_thread) {
        s_thread->Finish();
    }
}

void Wait ()
{
#ifdef AMREX_USE_MPI
    const int N = s_info.ispot;
    if (N > 0) {
        Vector<MPI_Request> reqs(N);
        Vector<MPI_Status> stats(N);
        for (int i = 0; i < N; ++i) {
            reqs[i] = ParallelDescriptor::Abarrier(s_comm).req();
        }
        ParallelDescriptor::Waitall(reqs, stats);
    }
#endif
}

void Notify ()
{
#ifdef AMREX_USE_MPI
    const int N = s_info.nspots - 1 - s_info.ispot;
    if (N > 0) {
        Vector<MPI_Request> reqs(N);
        Vector<MPI_Status> stats(N);
        for (int i = 0; i < N; ++i) {
            reqs[i] = ParallelDescriptor::Abarrier(s_comm).req();
        }
        ParallelDescriptor::Waitall(reqs, stats);
    }
#endif
}

}
