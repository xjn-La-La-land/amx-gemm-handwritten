#include "amx-gemm.hpp"

namespace amx {
using Kernel = GEMMKernelInt8;
using KernelMT = GEMMKernelInt8MT;


///////////////////////////////////////////////////////
// AMX GEMM compute kernel
///////////////////////////////////////////////////////

void Kernel::cpu_gemm_ref() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = C[OFFSET2D(i, j, ldc)];
            for (int k = 0; k < K; k++)
                sum += A[OFFSET2D(i, k, lda)] * B[OFFSET2D(k, j, ldb)];
            C[OFFSET2D(i, j, ldc)] = sum;
        }
    }
}

// 合法 (kernel, loop_order) routine 变体的单一真源(9 行)。成员指针在此(类内作用域)形成合法。
// pack 列驱动 buffer 分配; autotune 逐行枚举这 9 个合法组合。GEMM_/GEPP_ 各占两行(KN/KM)——
// 具体走哪行由 select_kernel 按形状(M≥N)决定, 那是 heuristic 策略, 不编码进本表。
const std::array<Kernel::KernelSpec, 9>& Kernel::kernel_specs() {
    // clang-format off
    static const std::array<KernelSpec, 9> table = {{
        /* kernel          loop_order            name      pA     pB    pC */
        { &Kernel::GEMM_,  LoopOrder::KN_GEPB,   "GEMM",   true,  true, true  },
        { &Kernel::GEMM_,  LoopOrder::KM_GEBP,   "GEMM",   true,  true, true  },
        { &Kernel::GEPP_,  LoopOrder::KN_GEPB,   "GEPP",   true,  true, false },
        { &Kernel::GEPP_,  LoopOrder::KM_GEBP,   "GEPP",   true,  true, false },
        { &Kernel::GEMP_,  LoopOrder::KN_GEPB,   "GEMP",   false, true, true  },
        { &Kernel::GEPB_,  LoopOrder::KN_GEPB,   "GEPB",   false, true, false },
        { &Kernel::GEPM_,  LoopOrder::KM_GEBP,   "GEPM",   true,  true, true  },
        { &Kernel::GEBP_,  LoopOrder::KM_GEBP,   "GEBP",   true,  true, false },
        { &Kernel::GEPDOT_,LoopOrder::MN_GEPDOT, "GEPDOT", false, true, true  },
    }};
    // clang-format on
    return table;
}

// Heuristic selection of different kernel routines based on matrix shapes.
void Kernel::select_kernel() {
    // 查 spec 行: 传 loop_order 则精确匹配 (kernel, loop_order); 省略(NONE)则按 kernel 取首行
    // (单 loop order 的 kernel 只有一行, 即其唯一行; 表中无 loop_order==NONE, 故 NONE 可当通配)。
    // 非法/不存在 -> nullptr。
    auto spec_of = [](FuncPtr k, LoopOrder lo = LoopOrder::NONE) -> const KernelSpec * {
        for (const auto &s : kernel_specs())
            if (s.kernel == k && (lo == LoopOrder::NONE || s.loop_order == lo))
                return &s;
        return nullptr;
    };

    if (params.kernel != nullptr && params.loop_order != LoopOrder::NONE) {
        // manual: 精确匹配用户指定的 (kernel, loop_order); 非法组合 -> nullptr -> 下方 abort。
        selected_spec_ = spec_of(params.kernel, params.loop_order);
    } else {
        // auto select: 形状 magic → kernel
        // clang-format off
        static const FuncPtr SHAPE_TABLE[8] = {
            /* 000 small M,N,K */ &Kernel::GEPB_,
            /* 001 large K     */ &Kernel::GEPDOT_,
            /* 010 large N     */ &Kernel::GEBP_,
            /* 011 large N,K   */ &Kernel::GEPM_,
            /* 100 large M     */ &Kernel::GEPB_,
            /* 101 large M,K   */ &Kernel::GEMP_,
            /* 110 large M,N   */ &Kernel::GEPP_,
            /* 111 large M,N,K */ &Kernel::GEMM_,
        };
        // clang-format on
        uint8_t magic = large_enough_m() << 2 | large_enough_n() << 1 | large_enough_k();
        FuncPtr kernel = SHAPE_TABLE[magic];

        if (kernel == &Kernel::GEMM_ || kernel == &Kernel::GEPP_)
            selected_spec_ = spec_of(kernel, (M >= N) ? LoopOrder::KN_GEPB : LoopOrder::KM_GEBP);
        else
            selected_spec_ = spec_of(kernel);
    }

    if (!selected_spec_) {
        std::cerr << "[Error] select_kernel: no matching kernel spec\n";
        std::abort();
    }
    // 分配 buffer 空间(loop order 与 pack 标志均来自 selected_spec_)
    switch (selected_spec_->loop_order) {
    case LoopOrder::KN_GEPB: // R1: panel A(M×KC) / block B(KC×NC) / matrix C
        if (selected_spec_->packA) bufA = Buffer<int8_t >::allocate(size_t(M)  * KC);
        if (selected_spec_->packB) bufB = Buffer<int8_t >::allocate(size_t(KC) * NC);
        if (selected_spec_->packC) bufC = Buffer<int32_t>::allocate(size_t(M)  * N);
        break;
    case LoopOrder::KM_GEBP: // R2: block A(MC×KC) / panel B(KC×N) / matrix C
        if (selected_spec_->packA) bufA = Buffer<int8_t >::allocate(size_t(MC) * KC);
        if (selected_spec_->packB) bufB = Buffer<int8_t >::allocate(size_t(KC) * N);
        if (selected_spec_->packC) bufC = Buffer<int32_t>::allocate(size_t(M)  * N);
        break;
    case LoopOrder::MN_GEPDOT: // R3: whole B(K×N) / matrix C; A 走 strided 不 pack
        if (selected_spec_->packB) bufB = Buffer<int8_t >::allocate(size_t(K) * N);
        if (selected_spec_->packC) bufC = Buffer<int32_t>::allocate(size_t(M) * N);
        break;
    case LoopOrder::NONE:
        break;
    }
}


void Kernel::pack_A() {
    if (loop_order() == LoopOrder::MN_GEPDOT) {
        make_view(Role::A, bufA, {M, K}).pack_from(A, lda); // 整块 2D
        return;
    }
    // KN_GEPB / KM_GEBP: K 方向按 KC 分块,每块 M×curKC(RowGrid)
    int off = 0;
    for (int kc = 0; kc < K; kc += KC) {
        int cur = min(KC, K - kc);
        make_view(Role::A, bufA, {M, cur}, off).pack_from(&A[OFFSET2D(0, kc, lda)], lda);
        off += M * cur;
    }
}

void Kernel::pack_B() {
    if (loop_order() == LoopOrder::MN_GEPDOT) {
        make_view(Role::B, bufB, {K, N}).pack_from(B, ldb); // 整块 2D
        return;
    }
    // KN_GEPB / KM_GEBP: K 方向按 KC 分块,每块 curKC×N(ColGrid)
    int off = 0;
    for (int kc = 0; kc < K; kc += KC) {
        int cur = min(KC, K - kc);
        make_view(Role::B, bufB, {cur, N}, off).pack_from(&B[OFFSET2D(kc, 0, ldb)], ldb);
        off += cur * N;
    }
}

void Kernel::unpack_C() {
    bool acc = params.beta != 0.0f;
    if (loop_order() == LoopOrder::KN_GEPB) { // 沿 N 分块,panel M×NC(RowGrid)
        int off = 0;
        for (int nc = 0; nc < N; nc += NC) {
            int cur = min(NC, N - nc);
            make_view(Role::C, bufC, {M, cur}, Strides{cur, MR}, off)
                .unpack_to(&C[OFFSET2D(0, nc, ldc)], ldc, acc);
            off += M * cur;
        }
    } else if (loop_order() == LoopOrder::KM_GEBP) { // 沿 M 分块,panel MC×N(ColGrid)
        int off = 0;
        for (int mc = 0; mc < M; mc += MC) {
            int cur = min(MC, M - mc);
            make_view(Role::C, bufC, {cur, N}, Strides{NR, cur}, off)
                .unpack_to(&C[OFFSET2D(mc, 0, ldc)], ldc, acc);
            off += cur * N;
        }
    } else if (loop_order() == LoopOrder::MN_GEPDOT) { // 整块,(NR,M)
        make_view(Role::C, bufC, {M, N}, Strides{NR, M}).unpack_to(C, ldc, acc);
    }
}

// 分配整矩阵 packed buffer(offline 路径:prepare_packed_data → GEMM_compute → restore_packed_data)
void Kernel::alloc_buffers() {
    bufA = Buffer<int8_t >::allocate(size_t(M) * K);
    bufB = Buffer<int8_t >::allocate(size_t(K) * N);
    bufC = Buffer<int32_t>::allocate(size_t(M) * N);
}

void Kernel::prepare_packed_data() {
    alloc_buffers();
    pack_A();
    pack_B();
}

// write back packed C matrix to original layout
void Kernel::restore_packed_data() {
    if (!bufA.valid() || !bufB.valid() || !bufC.valid()) {
        std::cerr << __func__ << ": Please call prepare_packed_data() first!\n";
        return;
    }
    unpack_C();
}

void Kernel::GEMM_compute() {
    if (!bufA.valid() || !bufB.valid() || !bufC.valid()) {
        std::cerr << __func__ << ": Please call prepare_packed_data() first!\n";
        return;
    }

    if (loop_order() == LoopOrder::KN_GEPB) {
        int off_A = 0, off_B = 0; // 元素偏移
        GEPBKernelConfig cfg;

        for (int kc = 0; kc < K; kc += KC) {
            int curKC = min(KC, K - kc);
            auto panelA = make_view(Role::A, bufA, {M, curKC}, off_A);

            int off_C = 0;
            cfg.overwrite_C = (kc == 0);

            for (int nc = 0; nc < N; nc += NC) {
                int curNC = min(NC, N - nc);
                auto blockB = make_view(Role::B, bufB, {curKC, curNC}, off_B);
                auto panelC = make_view(Role::C, bufC, {M, curNC}, Strides{curNC, MR}, off_C);
                GEPB_kernel(panelA, blockB, panelC, cfg);
                off_B += blockB.size();
                off_C += panelC.size();
            }
            off_A += panelA.size();
        }
    } else if (loop_order() == LoopOrder::KM_GEBP) {

        int off_A = 0, off_B = 0;
        for (int kc = 0; kc < K; kc += KC) {
            int curKC = min(KC, K - kc);
            auto panelB = make_view(Role::B, bufB, {curKC, N}, off_B);

            int off_C = 0;
            bool acc = kc != 0; // 首个 kc 覆盖 C,其余累加
            for (int mc = 0; mc < M; mc += MC) {
                int curMC = min(MC, M - mc);
                auto blockA = make_view(Role::A, bufA, {curMC, curKC}, off_A);
                auto panelC = make_view(Role::C, bufC, {curMC, N}, Strides{NR, curMC}, off_C);
                GEBP_kernel(blockA, panelB, panelC, acc); // compute kernel
                off_A += blockA.size();
                off_C += panelC.size();
            }
            off_B += panelB.size();
        }
    } else if (loop_order() == LoopOrder::MN_GEPDOT) {

        int32_t *C_ptr = bufC.data();
        auto vA = make_view(Role::A, bufA, {M, K}); // 整块 A(RowGrid)
        auto vB = make_view(Role::B, bufB, {K, N}); // 整块 B(ColGrid)
        for (int j = 0; j < N; j += NR) {
            const int8_t *B_panel_ptr = vB(0, j);

            for (int i = 0; i < M; i += MR) {
                const int8_t *B_ptr = B_panel_ptr;
                const int8_t *A_ptr = vA(i, 0);
                clear_4_tileC(); // clear tile C

                for (int k = 0; k < K; k += KR) {
                    load_2_tileA_l1(A_ptr); // load tile A0, A1
                    load_2_tileB_l2(B_ptr); // load tile B0, B1
                    run_4_tdp();            // compute tile C
                    B_ptr += NR * KR;
                    A_ptr += MR * KR;
                }

                store_4_tileC_l1(C_ptr); // store tile C
                C_ptr += MR * NR;
            }
        }
    }
}

void Kernel::GEMM() { (this->*selected_spec_->kernel)(); }

// large M, large N, large K
void Kernel::GEMM_() {
    if (loop_order() == LoopOrder::KN_GEPB) {
        assert(bufA.valid() && bufB.valid() && bufC.valid());
        GEPBKernelConfig cfg;

        for (int kc = 0; kc < K; kc += KC) {
            int curKC = min(KC, K - kc);
            auto panelA = make_view(
                Role::A, 
                bufA, 
                {M, curKC}
            );

            int off_C = 0;
            cfg.originA = &A[OFFSET2D(0, kc, lda)];
            cfg.overwrite_C = (kc == 0);

            for (int nc = 0; nc < N; nc += NC) {
                int curNC = min(NC, N - nc);
                auto blockB = make_view(
                    Role::B, 
                    bufB, 
                    {curKC, curNC}
                );
                blockB.pack_from(&B[OFFSET2D(kc, nc, ldb)], ldb); // pack block B
                auto panelC = make_view(
                    Role::C, 
                    bufC, 
                    {M, curNC}, 
                    Strides{curNC, MR}, 
                    off_C
                );

                cfg.fuse_packA = nc == 0;
                GEPB_kernel(panelA, blockB, panelC, cfg);

                off_C += panelC.size();
            }
        }
        unpack_C();
    }

    else if (loop_order() == LoopOrder::KM_GEBP) {
        assert(bufA.valid() && bufB.valid() && bufC.valid());

        for (int kc = 0; kc < K; kc += KC) {
            int curKC = min(KC, K - kc);
            auto panelB = make_view(
                Role::B, 
                bufB, 
                {curKC, N}
            );

            int off_C = 0;
            bool acc = kc != 0; // 首个 kc 覆盖 C,其余累加
            const int8_t *B_kc_base = &B[OFFSET2D(kc, 0, ldb)]; // base addr of panelB

            for (int mc = 0; mc < M; mc += MC) {
                int curMC = min(MC, M - mc);
                auto blockA = make_view(
                    Role::A, 
                    bufA, 
                    {curMC, curKC}
                );
                blockA.pack_from(&A[OFFSET2D(mc, kc, lda)], lda); // pack block A
                auto panelC = make_view(
                    Role::C, 
                    bufC, 
                    {curMC, N}, 
                    Strides{NR, curMC}, 
                    off_C
                );
                if (mc == 0) {
                    GEBP_kernel(blockA, panelB, panelC, acc, B_kc_base);
                } else {
                    GEBP_kernel(blockA, panelB, panelC, acc);
                }
                off_C += panelC.size();
            }
        }
        unpack_C();
    }

    else {
        std::cerr << "[Error] GEMM_: unsupported loop order: " << loop_order_str(loop_order()) << "\n";
        std::abort();
    }
}

// large M, large N, small K
void Kernel::GEPP_() {
    if (loop_order() == LoopOrder::KN_GEPB) {
        assert(bufA.valid() && bufB.valid());
        assert(!bufC.valid()); // no buffer for C

        GEPBKernelConfig cfg;

        for (int kc = 0; kc < K; kc += KC) {
            int curKC = min(KC, K - kc);
            auto panelA = make_view(
                Role::A, 
                bufA, 
                {M, curKC}
            );

            cfg.overwrite_C = (kc == 0) && (params.beta == 0.0f);
            cfg.originA = &A[OFFSET2D(0, kc, lda)];

            for (int nc = 0; nc < N; nc += NC) {
                int curNC = min(NC, N - nc);
                auto blockB = make_view(
                    Role::B, 
                    bufB, 
                    {curKC, curNC}
                );
                blockB.pack_from(&B[OFFSET2D(kc, nc, ldb)], ldb); // pack block B
                auto panelC = make_view(
                    Role::C, 
                    &C[OFFSET2D(0, nc, ldc)], 
                    {M, curNC}, 
                    ldc
                );

                cfg.fuse_packA = nc == 0;
                GEPB_kernel(panelA, blockB, panelC, cfg);
            }
        }

    }

    else if (loop_order() == LoopOrder::KM_GEBP) {
        assert(bufA.valid() && bufB.valid());
        assert(!bufC.valid()); // no buffer for C

        for (int kc = 0; kc < K; kc += KC) {
            int curKC = min(KC, K - kc);
            auto panelB = make_view(
                Role::B, 
                bufB, 
                {curKC, N}
            );

            const int8_t *B_kc_base = &B[OFFSET2D(kc, 0, ldb)]; // base addr of panelB
            for (int mc = 0; mc < M; mc += MC) {
                int curMC = min(MC, M - mc);
                auto blockA = make_view(
                    Role::A, 
                    bufA, 
                    {curMC, curKC}
                );
                blockA.pack_from(&A[OFFSET2D(mc, kc, lda)], lda); // pack block A
                auto panelC = make_view(
                    Role::C, 
                    &C[OFFSET2D(mc, 0, ldc)], 
                    {curMC, N}, 
                    ldc
                );
                if (mc == 0) {
                    GEBP_kernel(blockA, panelB, panelC, true, B_kc_base);
                } else {
                    GEBP_kernel(blockA, panelB, panelC, true);
                }
            }
        }
    }

    else {
        std::cerr << "[Error] GEPP_: unsupported loop order: " << loop_order_str(loop_order()) << "\n";
        std::abort();
    }
}

// large M, small N, large K
void Kernel::GEMP_() {
    assert(loop_order() == LoopOrder::KN_GEPB);
    assert(!bufA.valid());                // no buffer for A
    assert(bufB.valid() && bufC.valid()); // block B and matrix C are packed

    GEPBKernelConfig cfg;

    for (int kc = 0; kc < K; kc += KC) {
        int curKC = min(KC, K - kc);
        auto panelA = make_view(
            Role::A, 
            const_cast<int8_t *>(&A[OFFSET2D(0, kc, lda)]),
            {M, curKC}, 
            lda
        ); // strided A

        cfg.overwrite_C = kc == 0;
        int off_C = 0;

        for (int nc = 0; nc < N; nc += NC) {
            int curNC = min(NC, N - nc);
            auto blockB = make_view(
                Role::B, 
                bufB, 
                {curKC, curNC}
            );
            blockB.pack_from(&B[OFFSET2D(kc, nc, ldb)], ldb); // pack block B
            auto panelC = make_view(
                Role::C, 
                bufC, 
                {M, curNC}, 
                Strides{curNC, MR}, 
                off_C
            );

            GEPB_kernel(panelA, blockB, panelC, cfg); // 原来误传 bufA(无效),应为 panelA
            off_C += panelC.size();
        }
    }
    unpack_C();
}

// small M, large N, large K
void Kernel::GEPM_() {
    assert(loop_order() == LoopOrder::KM_GEBP);
    GEMM_();
}

// large M, small N, small K
void Kernel::GEPB_() {
    assert(loop_order() == LoopOrder::KN_GEPB);
    assert(!bufA.valid()); // no buffer for A
    assert(bufB.valid());  // block B is packed
    assert(!bufC.valid()); // no buffer for C

    GEPBKernelConfig cfg;

    for (int kc = 0; kc < K; kc += KC) {
        int curKC = min(KC, K - kc);
        auto panelA = make_view(
            Role::A, 
            const_cast<int8_t *>(&A[OFFSET2D(0, kc, lda)]),
            {M, curKC}, 
            lda
        ); // strided A

        cfg.overwrite_C = (kc == 0) && (params.beta == 0.0f);

        for (int nc = 0; nc < N; nc += NC) {
            int curNC = min(NC, N - nc);
            auto blockB = make_view(
                Role::B, 
                bufB, 
                {curKC, curNC}
            );
            blockB.pack_from(&B[OFFSET2D(kc, nc, ldb)], ldb); // pack block B
            auto panelC = make_view(
                Role::C, 
                &C[OFFSET2D(0, nc, ldc)], 
                {M, curNC}, 
                ldc
            );
            GEPB_kernel(panelA, blockB, panelC, cfg);
        }
    }

}

// small M, large N, small K
void Kernel::GEBP_() {
    assert(loop_order() == LoopOrder::KM_GEBP);
    GEPP_();
}

// small M, small N, large K
void Kernel::GEPDOT_() {
    assert(loop_order() == LoopOrder::MN_GEPDOT);
    assert(!bufA.valid());                // no buffer for A
    assert(bufB.valid() && bufC.valid()); // matrix B and matrix C are packed

    auto vB = make_view(Role::B, bufB, {K, N});
    vB.pack_from(B, ldb); // pack whole matrix B into packed buffer
    // A:strided 视图,用 tile() 寻址
    auto vA = make_view(Role::A, const_cast<int8_t *>(A), {M, K}, lda);
    // micro-kernel
    int32_t *C_ptr = bufC.data();
    for (int j = 0; j < N; j += NR) {
        const int8_t *B_panel_ptr = vB(0, j);

        for (int i = 0; i < M; i += MR) {
            const int8_t *B_ptr = B_panel_ptr;
            clear_4_tileC(); // clear tile C

            for (int k = 0; k < K; k += KR) {
                load_2_tileA_l1(vA.tile(i, k)); // load tile A0, A1
                load_2_tileB_l2(B_ptr);         // load tile B0, B1
                run_4_tdp();                    // compute tile C
                B_ptr += NR * KR;
            }

            store_4_tileC_l1(C_ptr); // store tile C
            C_ptr += MR * NR;
        }
    }

    auto vC = make_view(Role::C, bufC, {M, N}, Strides{NR, M}); // GEPDOT C: ColGrid
    vC.unpack_to(C, ldc, true); // C += packed result
}

template <bool fuse_packA, bool overwrite_C>
void Kernel::GEPB_kernel_impl_denseABC(View<int8_t> panelA,
                                       View<int8_t> blockB,
                                       View<int32_t> panelC,
                                       const int8_t *originA) {

    int32_t *C_ptr = panelC.data();
    SWPFHelper swpf_ctx_a(
        MR * panelA.cols() * sizeof(int8_t),
        2,          // step
        _MM_HINT_T1 // hint: prefetch to L2
    );
    SWPFHelper swpf_ctx_c(
        MR * NR * sizeof(int32_t),
        4,          // step
        _MM_HINT_T0 // hint: prefetch to L1
    );

    for (int i = 0; i < panelA.rows(); i += MR) {

        const int8_t *cur_stripA_ptr = panelA(i, 0);

        if constexpr (fuse_packA) { // pack stripA (MR x KC)
            int8_t *dst = const_cast<int8_t *>(cur_stripA_ptr);
            for (int k = 0; k < panelA.cols(); k += KR) {
                pack_2_tile_a(&originA[OFFSET2D(i, k, lda)], dst, lda);
            }
            swpf_ctx_a.on = false; // when packing into bufferA, no need to prefetch
        } else { // if A_kc_base is nullptr, it means the stripA is already packed,
                 // we just need to prefetch it
            const int8_t *next_stripA_ptr = panelA(i + MR, 0);
            swpf_ctx_a.init(next_stripA_ptr);
        }

        const int8_t *B_ptr = blockB.data();

        // GESB
        for (int j = 0; j < blockB.cols(); j += NR) {

            const int8_t *A_ptr = cur_stripA_ptr;
            if constexpr (overwrite_C) {
                clear_4_tileC();
            } else {
                load_4_tileC_l1(C_ptr);
            }

            const int8_t *next_C_ptr = reinterpret_cast<const int8_t *>(C_ptr + MR * NR);
            swpf_ctx_c.init(next_C_ptr);

            for (int k = 0; k < blockB.rows(); k += KR) {
                load_2_tileB_l2(B_ptr); // tileload B0, B1
                load_2_tileA_l1(A_ptr); // tileload A0, A1
                run_4_tdp();
                B_ptr += NR * KR;
                A_ptr += MR * KR;

                swpf_ctx_a.prefetch();
                swpf_ctx_c.prefetch();
            } // end for k loop

            store_4_tileC_l1(C_ptr);
            C_ptr += MR * NR;
        }
    }
}

template <bool fuse_packA, bool overwrite_C>
void Kernel::GEPB_kernel_impl_denseAB_stridedC(View<int8_t> panelA,
                                                View<int8_t> blockB,
                                                View<int32_t> panelC,
                                                const int8_t *originA) {

    SWPFHelper swpf_ctx_a(
        MR * panelA.cols() * sizeof(int8_t),
        2,          // step
        _MM_HINT_T1 // hint: prefetch to L2
    );
    SWPFHelper swpf_ctx_c(
        MR * NR * sizeof(int32_t),
        4, // step
        NR * sizeof(int32_t),
        panelC.row_stride() * sizeof(int32_t),
        _MM_HINT_ET0 // hint: prefetch to L1
    );

    for (int i = 0; i < panelA.rows(); i += MR) {

        const int8_t *cur_stripA_ptr = panelA(i, 0);

        if constexpr (fuse_packA) { // pack stripA (MR x KC)
            int8_t *dst = const_cast<int8_t *>(cur_stripA_ptr);
            for (int k = 0; k < panelA.cols(); k += KR) {
                pack_2_tile_a(&originA[OFFSET2D(i, k, lda)], dst, lda);
            }
            swpf_ctx_a.on = false; // when packing into bufferA, no need to prefetch
        } else { // if A_kc_base is nullptr, it means the stripA is already packed,
                          // we just need to prefetch it
            const int8_t *next_stripA_ptr = panelA(i + MR, 0);
            swpf_ctx_a.init(next_stripA_ptr);
        }

        const int8_t *B_ptr = blockB.data();

        // GESB
        for (int j = 0; j < blockB.cols(); j += NR) {

            const int8_t *A_ptr = cur_stripA_ptr;
            if constexpr (overwrite_C) {
                clear_4_tileC();
            } else {
                load_4_tileC_l1(panelC.tile(i, j));
            }

            // 光栅序下一个 C tile-group(预取用),原 next_tile_ptr 内联
            const bool same_row = (j + NR < panelC.cols());
            const int  ni = same_row ? i : i + MR;
            const int  nj = same_row ? j + NR : 0;
            const int8_t *next_C_ptr =
                    reinterpret_cast<const int8_t *>(panelC(ni, nj));
            swpf_ctx_c.init(next_C_ptr);

            for (int k = 0; k < blockB.rows(); k += KR) {
                load_2_tileB_l2(B_ptr); // tileload B0, B1
                load_2_tileA_l1(A_ptr); // tileload A0, A1
                run_4_tdp();
                B_ptr += NR * KR;
                A_ptr += MR * KR;

                swpf_ctx_a.prefetch();
                swpf_ctx_c.prefetch();
            } // end for k loop

            store_4_tileC_l1(panelC.tile(i, j));
        }
    }
}

template <bool overwrite_C>
void Kernel::GEPB_kernel_impl_denseBC_stridedA(View<int8_t> panelA,
                                                View<int8_t> blockB,
                                                View<int32_t> panelC) {

    int32_t *C_ptr = panelC.data();
    SWPFHelper swpf_ctx_a(MR * panelA.cols() * sizeof(int8_t),
                                                2, // step
                                                panelA.cols() * sizeof(int8_t),
                                                panelA.row_stride() * sizeof(int8_t),
                                                _MM_HINT_T1 // hint: prefetch to L2
    );
    SWPFHelper swpf_ctx_c(MR * NR * sizeof(int32_t),
                                                4,          // step
                                                _MM_HINT_T0 // hint: prefetch to L1
    );

    for (int i = 0; i < panelA.rows(); i += MR) {

        const int8_t *next_stripA_ptr = panelA(i + MR, 0);
        swpf_ctx_a.init(next_stripA_ptr);

        const int8_t *B_ptr = blockB.data();

        // GESB
        for (int j = 0; j < blockB.cols(); j += NR) {

            if constexpr (overwrite_C) {
                clear_4_tileC();
            } else {
                load_4_tileC_l1(C_ptr);
            }

            const int8_t *next_C_ptr =
                    reinterpret_cast<const int8_t *>(C_ptr + MR * NR);
            swpf_ctx_c.init(next_C_ptr);

            for (int k = 0; k < blockB.rows(); k += KR) {
                load_2_tileB_l2(B_ptr);             // tileload B0, B1
                load_2_tileA_l1(panelA.tile(i, k)); // tileload A0, A1
                run_4_tdp();
                B_ptr += NR * KR;

                swpf_ctx_a.prefetch();
                swpf_ctx_c.prefetch();
            } // end for k loop

            store_4_tileC_l1(C_ptr);
            C_ptr += MR * NR;
        }
    }
}

template <bool overwrite_C>
void Kernel::GEPB_kernel_impl_denseB_stridedAC(View<int8_t> panelA,
                                               View<int8_t> blockB,
                                               View<int32_t> panelC) {

    SWPFHelper swpf_ctx_a(MR * panelA.cols() * sizeof(int8_t),
                                                2, // step
                                                panelA.cols() * sizeof(int8_t),
                                                panelA.row_stride() * sizeof(int8_t),
                                                _MM_HINT_T1 // hint: prefetch to L2
    );
    SWPFHelper swpf_ctx_c(MR * NR * sizeof(int32_t),
                                                4, // step
                                                NR * sizeof(int32_t), panelC.row_stride() * sizeof(int32_t),
                                                _MM_HINT_T0 // hint: prefetch to L1
    );

    for (int i = 0; i < panelA.rows(); i += MR) {

        const int8_t *next_stripA_ptr = panelA(i + MR, 0);
        swpf_ctx_a.init(next_stripA_ptr);

        const int8_t *B_ptr = blockB.data();

        // GESB
        for (int j = 0; j < blockB.cols(); j += NR) {

            if constexpr (overwrite_C) {
                clear_4_tileC();
            } else {
                load_4_tileC_l1(panelC.tile(i, j));
            }

            // 光栅序下一个 C tile-group(预取用),原 next_tile_ptr 内联
            const bool same_row = (j + NR < panelC.cols());
            const int  ni = same_row ? i : i + MR;
            const int  nj = same_row ? j + NR : 0;
            const int8_t *next_C_ptr =
                    reinterpret_cast<const int8_t *>(panelC(ni, nj));
            swpf_ctx_c.init(next_C_ptr);

            for (int k = 0; k < blockB.rows(); k += KR) {
                load_2_tileB_l2(B_ptr);             // tileload B0, B1
                load_2_tileA_l1(panelA.tile(i, k)); // tileload A0, A1
                run_4_tdp();
                B_ptr += NR * KR;

                swpf_ctx_a.prefetch();
                swpf_ctx_c.prefetch();
            } // end for k loop

            store_4_tileC_l1(panelC.tile(i, j));
        }
    }
}

void Kernel::GEPB_kernel(View<int8_t> panelA, View<int8_t> blockB,
                         View<int32_t> panelC,
                         const GEPBKernelConfig &cfg) {
    using DenseABCFn = void (Kernel::*)(View<int8_t> , View<int8_t> ,
                                        View<int32_t> , const int8_t*);
    static constexpr DenseABCFn dense_abc_dispatch[] = {
            &Kernel::GEPB_kernel_impl_denseABC<false, false>,
            &Kernel::GEPB_kernel_impl_denseABC<false, true>,
            &Kernel::GEPB_kernel_impl_denseABC<true, false>,
            &Kernel::GEPB_kernel_impl_denseABC<true, true>,
    };

    using DenseABStridedCFn = void (Kernel::*)(
            View<int8_t> , View<int8_t> , View<int32_t> , const int8_t *);
    static constexpr DenseABStridedCFn dense_ab_strided_c_dispatch[] = {
            &Kernel::GEPB_kernel_impl_denseAB_stridedC<false, false>,
            &Kernel::GEPB_kernel_impl_denseAB_stridedC<false, true>,
            &Kernel::GEPB_kernel_impl_denseAB_stridedC<true, false>,
            &Kernel::GEPB_kernel_impl_denseAB_stridedC<true, true>,
    };

    using DenseBCStridedAFn = void (Kernel::*)(
            View<int8_t> , View<int8_t> , View<int32_t> );
    static constexpr DenseBCStridedAFn dense_bc_strided_a_dispatch[] = {
            &Kernel::GEPB_kernel_impl_denseBC_stridedA<false>,
            &Kernel::GEPB_kernel_impl_denseBC_stridedA<true>,
    };

    using DenseBStridedACFn = void (Kernel::*)(
            View<int8_t> , View<int8_t> , View<int32_t> );
    static constexpr DenseBStridedACFn dense_b_strided_ac_dispatch[] = {
            &Kernel::GEPB_kernel_impl_denseB_stridedAC<false>,
            &Kernel::GEPB_kernel_impl_denseB_stridedAC<true>,
    };

    check_shape_health(panelA, blockB, panelC);

    assert(blockB.is_dense() && "GEPB_kernel requires dense layout for blockB");
    if (panelA.is_dense() && panelC.is_dense()) {
        (this->*dense_abc_dispatch[cfg.dense_ab_index()])(
                panelA, blockB, panelC, cfg.originA);
    } else if (panelA.is_dense() && panelC.is_strided()) {
        (this->*dense_ab_strided_c_dispatch[cfg.dense_ab_index()])(
                panelA, blockB, panelC, cfg.originA);
    } else if (panelA.is_strided() && panelC.is_dense()) {
        (this->*dense_bc_strided_a_dispatch[cfg.overwrite_c_index()])(
                panelA, blockB, panelC);
    } else {
        (this->*dense_b_strided_ac_dispatch[cfg.overwrite_c_index()])(
                panelA, blockB, panelC);
    }
}

template <>
void Kernel::GEBP_kernel_impl<true, true>(View<int8_t> blockA,
                                          View<int8_t> panelB,
                                          View<int32_t> panelC, bool acc,
                                          const int8_t *B_kc_base) {

    int32_t *C_ptr = panelC.data();
    SWPFHelper swpf_ctx_b(
        panelB.rows() * NR * sizeof(int8_t), 
        2, 
        _MM_HINT_T1
    );
    SWPFHelper swpf_ctx_c(
        MR * NR * sizeof(int32_t), 
        4, 
        _MM_HINT_T0
    );

    for (int j = 0; j < panelB.cols(); j += NR) {

        const int8_t *cur_stripB_ptr = panelB(0, j);
        if (B_kc_base != nullptr) { // pack stripB (KC x NR)
            int8_t *dst = const_cast<int8_t *>(cur_stripB_ptr);
            for (int k = 0; k < panelB.rows(); k += KR) {
                pack_2_tile_b(&B_kc_base[OFFSET2D(k, j, ldb)], dst, ldb);
            }
        } else {
            const int8_t *next_stripB_ptr = panelB(0, j + NR);
            swpf_ctx_b.init(next_stripB_ptr);
        }

        const int8_t *A_ptr = blockA.data();

        // GEBS
        for (int i = 0; i < blockA.rows(); i += MR) {

            const int8_t *B_ptr = cur_stripB_ptr;
            if (acc)
                load_4_tileC_l1(C_ptr);
            else
                clear_4_tileC();

            const int8_t *next_C_ptr =
                    reinterpret_cast<const int8_t *>(C_ptr + MR * NR);
            swpf_ctx_c.init(next_C_ptr);

            for (int k = 0; k < blockA.cols(); k += KR) {
                load_2_tileA_l2(A_ptr); // tileload A0, A1
                load_2_tileB_l1(B_ptr); // tileload B0, B1
                run_4_tdp();
                A_ptr += MR * KR;
                B_ptr += NR * KR;

                swpf_ctx_b.prefetch();
                swpf_ctx_c.prefetch();
            } // end for k loop

            store_4_tileC_l1(C_ptr);
            C_ptr += MR * NR;
        }
    }
}

template <>
void Kernel::GEBP_kernel_impl<true, false>(View<int8_t> blockA,
                                           View<int8_t> panelB,
                                           View<int32_t> panelC, bool acc,
                                           const int8_t *B_kc_base) {

    SWPFHelper swpf_ctx_b(
        panelB.rows() * NR * sizeof(int8_t), 
        2, 
        _MM_HINT_T1
    );
    SWPFHelper swpf_ctx_c(
        MR * NR * sizeof(int32_t), 
        4, 
        NR * sizeof(int32_t),
        panelC.row_stride() * sizeof(int32_t), 
        _MM_HINT_T0
    );

    for (int j = 0; j < panelB.cols(); j += NR) {

        const int8_t *cur_stripB_ptr = panelB(0, j);
        if (B_kc_base != nullptr) { // pack stripB (KC x NR)
            int8_t *dst = const_cast<int8_t *>(cur_stripB_ptr);
            for (int k = 0; k < panelB.rows(); k += KR) {
                pack_2_tile_b(&B_kc_base[OFFSET2D(k, j, ldb)], dst, ldb);
            }
        } else {
            const int8_t *next_stripB_ptr = panelB(0, j + NR);
            swpf_ctx_b.init(next_stripB_ptr);
        }

        const int8_t *A_ptr = blockA.data();

        // GEBS
        for (int i = 0; i < blockA.rows(); i += MR) {

            const int8_t *B_ptr = cur_stripB_ptr;
            if (acc)
                load_4_tileC_l1(panelC.tile(i, j));
            else
                clear_4_tileC();

            // 光栅序下一个 C tile-group(预取用),原 next_tile_ptr 内联
            const bool same_row = (j + NR < panelC.cols());
            const int  ni = same_row ? i : i + MR;
            const int  nj = same_row ? j + NR : 0;
            const int8_t *next_C_ptr =
                    reinterpret_cast<const int8_t *>(panelC(ni, nj));
            swpf_ctx_c.init(next_C_ptr);

            for (int k = 0; k < blockA.cols(); k += KR) {
                load_2_tileA_l2(A_ptr); // tileload A0, A1
                load_2_tileB_l1(B_ptr); // tileload B0, B1
                run_4_tdp();
                A_ptr += MR * KR;
                B_ptr += NR * KR;

                swpf_ctx_b.prefetch();
                swpf_ctx_c.prefetch();
            } // end for k loop

            store_4_tileC_l1(panelC.tile(i, j));
        }
    }
}

void Kernel::GEBP_kernel(View<int8_t> blockA, View<int8_t> panelB,
                         View<int32_t> panelC, bool acc,
                         const int8_t *B_kc_base) {
    check_shape_health(blockA, panelB, panelC);

    using Impl = void (Kernel::*)(View<int8_t> , View<int8_t> ,
                                  View<int32_t> , bool, const int8_t *);
    Impl impl;

    assert(blockA.is_dense() && "GEBP_kernel requires dense layout for blockA");
    assert(panelB.is_dense() && "GEBP_kernel requires dense layout for panelB");

    if (panelC.is_dense()) {
        impl = &Kernel::GEBP_kernel_impl<true, true>;
    } else {
        impl = &Kernel::GEBP_kernel_impl<true, false>;
    }

    (this->*impl)(blockA, panelB, panelC, acc, B_kc_base);
}

///////////////////////////////////////////////////////
// Multi-threaded Kernel Management
///////////////////////////////////////////////////////

// 找到每个线程对应的 Kernel 实例并初始化
void KernelMT::init_kernel_per_thread(int tid, int core_id) {
    try {
        bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
    } catch (const std::exception &e) {
        std::cerr << "Thread bind failed: " << e.what() << std::endl;
        return;
    }

    int blocks_m = ceil_div(M, MC);
    int blocks_n = ceil_div(N, NC);
    int total_blocks = blocks_m * blocks_n;
    int num_threads = params.core_list.size();

    // 使用简单的 Round-Robin 分配任务块
    for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
        int bm = (block_id / blocks_n) * MC;
        int bn = (block_id % blocks_n) * NC;

        // 在 Kernel_pool 中创建 Kernel 实例
        auto kernel_ptr = std::make_unique<Kernel>(
                min(MC, M - bm), min(NC, N - bn), K, lda, ldb, ldc,
                &A[OFFSET2D(bm, 0, lda)], &B[OFFSET2D(0, bn, ldb)],
                &C[OFFSET2D(bm, bn, ldc)], gemm_params);

        kernel_pool[block_id] = std::move(kernel_ptr);
    }
}

void KernelMT::init_kernels() {
    int num_threads = params.core_list.size();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int blocks_m = ceil_div(M, MC);
    int blocks_n = ceil_div(N, NC);
    int total_blocks = blocks_m * blocks_n;
    kernel_pool.resize(
            total_blocks); // 调整 kernel_pool 大小以容纳所有线程的 Kernel 实例

    // 启动线程
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&KernelMT::init_kernel_per_thread, this, i,
                                                  params.core_list[i]);
    }

    // 等待所有线程完成
    for (auto &t : threads) {
        if (t.joinable())
            t.join();
    }
}

void KernelMT::GEMM_per_thread(int tid, int core_id) {
    try {
        bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
    } catch (const std::exception &e) {
        std::cerr << "Thread bind failed: " << e.what() << std::endl;
        return;
    }

    Kernel::amx_init(); // 初始化 AMX

    // 找到对应的 Kernel 实例
    int num_threads = params.core_list.size();
    int total_blocks = kernel_pool.size();
    for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
        auto &kernel = kernel_pool[block_id];
        if (kernel) {
            kernel->GEMM();
        }
    }
}

void KernelMT::GEMM() {
    int num_threads = params.core_list.size();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 启动线程
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&KernelMT::GEMM_per_thread, this, i,
                                                  params.core_list[i]);
    }

    // 等待所有线程完成
    for (auto &t : threads) {
        if (t.joinable())
            t.join();
    }
}

void KernelMT::prepare_packed_data_per_thread(int tid, int core_id) {
    try {
        bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
    } catch (const std::exception &e) {
        std::cerr << "Thread bind failed: " << e.what() << std::endl;
        return;
    }

    // 找到对应的 Kernel 实例
    int num_threads = params.core_list.size();
    int total_blocks = kernel_pool.size();
    for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
        auto &kernel = kernel_pool[block_id];
        if (kernel) {
            kernel->prepare_packed_data();
        }
    }
}

void KernelMT::prepare_packed_data() {
    int num_threads = params.core_list.size();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 启动线程
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&KernelMT::prepare_packed_data_per_thread, this, i,
                                                  params.core_list[i]);
    }

    // 等待所有线程完成
    for (auto &t : threads) {
        if (t.joinable())
            t.join();
    }
}

void KernelMT::GEMM_compute_per_thread(int tid, int core_id) {
    try {
        bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
    } catch (const std::exception &e) {
        std::cerr << "Thread bind failed: " << e.what() << std::endl;
        return;
    }

    Kernel::amx_init(); // 初始化 AMX

    // 找到对应的 Kernel 实例
    int num_threads = params.core_list.size();
    int total_blocks = kernel_pool.size();
    for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
        auto &kernel = kernel_pool[block_id];
        if (kernel) {
            kernel->GEMM_compute();
        }
    }
}

void KernelMT::GEMM_compute() {
    int num_threads = params.core_list.size();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 启动线程
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&KernelMT::GEMM_compute_per_thread, this, i,
                                                  params.core_list[i]);
    }

    // 等待所有线程完成
    for (auto &t : threads) {
        if (t.joinable())
            t.join();
    }
}

void KernelMT::restore_packed_data_per_thread(int tid, int core_id) {
    try {
        bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
    } catch (const std::exception &e) {
        std::cerr << "Thread bind failed: " << e.what() << std::endl;
        return;
    }

    // 找到对应的 Kernel 实例
    int num_threads = params.core_list.size();
    int total_blocks = kernel_pool.size();
    for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
        auto &kernel = kernel_pool[block_id];
        if (kernel) {
            kernel->restore_packed_data();
        }
    }
}

void KernelMT::restore_packed_data() {
    int num_threads = params.core_list.size();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 启动线程
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&KernelMT::restore_packed_data_per_thread, this, i,
                                                  params.core_list[i]);
    }

    // 等待所有线程完成
    for (auto &t : threads) {
        if (t.joinable())
            t.join();
    }
}

} // namespace amx
