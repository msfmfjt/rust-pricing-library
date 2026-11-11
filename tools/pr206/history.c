// Diagnostic-only dyld interposition for the five allocator imports of the CI wheel.
// Record metadata in preallocated shared mappings. Never read user allocations.
#include <crt_externs.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef uint64_t word;
typedef _Atomic(uint64_t) atomic_word;
enum { ALL = 1 << 18, NATIVE = 1 << 18, SAMPLED = 1 << 16, WORDS = 8 };
enum { MALLOC_RETURN = 1, CALLOC_RETURN, ALIGNED_RETURN,
       FREE_CALL, FREE_RETURN, REALLOC_CALL, REALLOC_RETURN };
struct row { atomic_word v[WORDS]; };
struct trace {
    atomic_word magic, version, pid, all_count, native_count, sampled_count, wheel_lo, wheel_hi;
    struct row all[ALL], native[NATIVE], sampled[SAMPLED];
};
_Static_assert(sizeof(struct row) == 64, "file format");
_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2, "signal-safe 64-bit atomics required");
static _Atomic(struct trace *) active;
static int image_fd = -1;
static pid_t owner_pid;
static unsigned address_mask;

static void map_image(const struct mach_header *header, intptr_t slide) {
    Dl_info image;
    if (header->magic != MH_MAGIC_64 || !dladdr(header, &image) || !image.dli_fname) return;
    const struct mach_header_64 *mh = (const void *)header;
    const struct load_command *lc = (const void *)(mh + 1);
    for (uint32_t i = 0; i < mh->ncmds; i++, lc = (const void *)((const char *)lc + lc->cmdsize)) {
        if (lc->cmd != LC_SEGMENT_64) continue;
        const struct segment_command_64 *seg = (const void *)lc;
        if (strncmp(seg->segname, "__TEXT", sizeof(seg->segname))) continue;
        word lo = seg->vmaddr + slide, hi = lo + seg->vmsize;
        char line[2048];
        int n = snprintf(line, sizeof(line), "%llx %llx %llx %s\n",
                         (unsigned long long)lo, (unsigned long long)hi,
                         (unsigned long long)seg->vmaddr, image.dli_fname);
        if (n > 0 && n < (int)sizeof(line)) (void)write(image_fd, line, (size_t)n);
        if (strstr(image.dli_fname, "/rust_pricing/") && strstr(image.dli_fname, ".so")) {
            struct trace *t = atomic_load(&active);
            if (t) { atomic_store(&t->wheel_lo, lo); atomic_store(&t->wheel_hi, hi); }
        }
        break;
    }
}

static void after_fork(void) { atomic_store(&active, NULL); }

__attribute__((constructor)) static void start(void) {
    const char *dir = getenv("PR206_EVENT_DIR"), *marker = getenv("PR206_EVENT_ARG");
    if (!dir || !marker) return;
    char **args = *_NSGetArgv();
    int found = 0;
    for (int i = 1; i < *_NSGetArgc(); i++) if (!strcmp(args[i], marker)) found = 1;
    if (!found) return;
    owner_pid = getpid();
    const char *sampling = getenv("PR206_ADDRESS_COHORT");
    address_mask = sampling && !strcmp(sampling, "1-in-128") ? 127 : 0;
    char file[4096];
    int fd = -1;
    for (unsigned i = 0; i < 8; i++) {
        int n = snprintf(file, sizeof(file), "%s/events-%d-%u.bin", dir, owner_pid, i);
        if (n < 0 || n >= (int)sizeof(file) - 5) return;
        fd = open(file, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd >= 0 || errno != EEXIST) break;
    }
    if (fd < 0 || ftruncate(fd, sizeof(struct trace)) < 0) return;
    struct trace *t = mmap(NULL, sizeof(*t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (t == MAP_FAILED) return;
    strcat(file, ".maps");
    image_fd = open(file, O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, 0600);
    if (image_fd < 0) return;
    t->version = 1;
    t->pid = owner_pid;
    t->magic = 0x5052323036455654ULL;
    if (pthread_atfork(NULL, NULL, after_fork) != 0) return;
    atomic_store(&active, t);
    _dyld_register_func_for_add_image(map_image);
}
static inline word mix(word v) {
    v ^= v >> 33; v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ULL;
    return v ^ (v >> 33);
}
static inline void store_row(struct row *ring, word capacity, word index, const word data[WORDS]) {
    struct row *r = ring + ((index - 1) & (capacity - 1));
    const word busy = 1ULL << 63;
    word previous = atomic_load_explicit(&r->v[0], memory_order_relaxed);
    if (previous >= index || !atomic_compare_exchange_strong_explicit(
        &r->v[0], &previous, busy | index, memory_order_acq_rel, memory_order_relaxed)) return;
    for (unsigned i = 1; i < WORDS; i++) atomic_store_explicit(&r->v[i], data[i], memory_order_relaxed);
    atomic_store_explicit(&r->v[0], index, memory_order_release);
}
static inline word event(word op, word ptr, word aux, word size, word pc) {
    struct trace *t = atomic_load_explicit(&active, memory_order_acquire);
    if (!t) return 0;
    // Apply the same stable hash to malloc and free. Record a realloc return
    // for a selected old address even when the new address is outside the cohort.
    if (address_mask && (!ptr || (mix(ptr) & address_mask)) &&
        !(op == REALLOC_RETURN && aux)) return 0;
    word seq = atomic_fetch_add_explicit(&t->all_count, 1, memory_order_relaxed) + 1;
    word row[] = {0, seq, op, (uintptr_t)pthread_self(), ptr, aux, size, pc};
    store_row(t->all, ALL, seq, row);
    word lo = atomic_load_explicit(&t->wheel_lo, memory_order_relaxed);
    if (lo && pc >= lo && pc < atomic_load_explicit(&t->wheel_hi, memory_order_relaxed)) {
        word idx = atomic_fetch_add_explicit(&t->native_count, 1, memory_order_relaxed) + 1;
        store_row(t->native, NATIVE, idx, row);
    }
    if (ptr && !(mix(ptr) & 1023)) {
        word idx = atomic_fetch_add_explicit(&t->sampled_count, 1, memory_order_relaxed) + 1;
        store_row(t->sampled, SAMPLED, idx, row);
    }
    return seq;
}
#define CALLER ((uintptr_t)__builtin_extract_return_addr(__builtin_return_address(0)))
#define INTERPOSE(repl, original) \
    __attribute__((used, section("__DATA,__interpose"))) static const struct { \
        const void *replacement; const void *replacee; \
    } pair_##repl = {(const void *)repl, (const void *)original}
static __attribute__((noinline)) void *record_malloc(size_t n) {
    void *p = malloc(n); int saved = errno;
    event(MALLOC_RETURN, (uintptr_t)p, 0, n, CALLER); errno = saved; return p;
}
static __attribute__((noinline)) void *record_calloc(size_t n, size_t width) {
    void *p = calloc(n, width); int saved = errno;
    event(CALLOC_RETURN, (uintptr_t)p, n, n * width, CALLER); errno = saved; return p;
}
static __attribute__((noinline)) int record_aligned(void **out, size_t align, size_t n) {
    int rc = posix_memalign(out, align, n); int saved = errno;
    event(ALIGNED_RETURN, rc ? 0 : (uintptr_t)*out, align, n, CALLER); errno = saved; return rc;
}
static __attribute__((noinline)) void record_free(void *p) {
    word addr = (uintptr_t)p, pc = CALLER;
    int saved = errno; word call = event(FREE_CALL, addr, 0, 0, pc); errno = saved;
    free(p); saved = errno; event(FREE_RETURN, addr, call, 0, pc); errno = saved;
}
static __attribute__((noinline)) void *record_realloc(void *p, size_t n) {
    word addr = (uintptr_t)p, pc = CALLER;
    int saved = errno; word call = event(REALLOC_CALL, addr, 0, n, pc); errno = saved;
    void *q = realloc(p, n); saved = errno;
    event(REALLOC_RETURN, (uintptr_t)q, call, n, pc); errno = saved; return q;
}
INTERPOSE(record_malloc, malloc);
INTERPOSE(record_calloc, calloc);
INTERPOSE(record_aligned, posix_memalign);
INTERPOSE(record_free, free);
INTERPOSE(record_realloc, realloc);
