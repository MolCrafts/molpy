// GoogleTest suite for molc C API.
//
// Build:
//   cargo build -p molcrafts-molc          # build Rust library first
//   cmake -B build tests/cpp && cmake --build build && ctest --test-dir build

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <string>

extern "C" {
#include "molc.h"
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define ASSERT_MOLRS_OK(expr)                                                  \
    do {                                                                        \
        MolcStatus s_ = (expr);                                                \
        ASSERT_EQ(s_, MOLC_STATUS_OK)                                          \
            << "molrs error: " << molc_last_error();                           \
    } while (0)

#define EXPECT_MOLRS_OK(expr)                                                  \
    do {                                                                        \
        MolcStatus s_ = (expr);                                                \
        EXPECT_EQ(s_, MOLC_STATUS_OK)                                          \
            << "molrs error: " << molc_last_error();                           \
    } while (0)

static uint32_t intern(const char* key) {
    uint32_t id = 0;
    MolcStatus s = molc_intern_key(key, &id);
    if (s != MOLC_STATUS_OK) {
        ADD_FAILURE() << "intern(\"" << key << "\") failed: " << molc_last_error();
    }
    return id;
}

// ---------------------------------------------------------------------------
// Fixture: manages init/shutdown per test suite
// ---------------------------------------------------------------------------

class MolcTest : public ::testing::Test {
protected:
    void SetUp() override { molc_init(); }
};

// ===== Lifecycle & Utilities ==============================================

TEST_F(MolcTest, InitShutdown) {
    // init already called in SetUp
    molc_shutdown();
    // re-init so TearDown doesn't break
    molc_init();
}

TEST_F(MolcTest, InternKeyRoundtrip) {
    uint32_t id1 = intern("gtest_key_a");
    uint32_t id2 = intern("gtest_key_a");
    EXPECT_EQ(id1, id2) << "same string must yield same id";

    uint32_t id3 = intern("gtest_key_b");
    EXPECT_NE(id1, id3) << "different strings must yield different ids";

    const char* name = molc_key_name(id1);
    ASSERT_NE(name, nullptr);
    EXPECT_STREQ(name, "gtest_key_a");
}

TEST_F(MolcTest, LastErrorAfterBadDrop) {
    MolcFrameHandle bogus{999, 999};
    MolcStatus s = molc_frame_drop(bogus);
    EXPECT_NE(s, MOLC_STATUS_OK);
    const char* msg = molc_last_error();
    ASSERT_NE(msg, nullptr);
    EXPECT_GT(std::strlen(msg), 0u);
}

// ===== Frame ==============================================================

TEST_F(MolcTest, FrameLifecycle) {
    MolcFrameHandle frame{};
    ASSERT_MOLRS_OK(molc_frame_new(&frame));

    MolcFrameHandle clone{};
    ASSERT_MOLRS_OK(molc_frame_clone(frame, &clone));

    ASSERT_MOLRS_OK(molc_frame_drop(frame));
    ASSERT_MOLRS_OK(molc_frame_drop(clone));

    // double drop must fail
    EXPECT_NE(molc_frame_drop(frame), MOLC_STATUS_OK);
}

TEST_F(MolcTest, FrameMetadata) {
    MolcFrameHandle frame{};
    ASSERT_MOLRS_OK(molc_frame_new(&frame));

    MolcMetaValue value{};
    value.dtype = MOLC_META_TYPE_STRING;
    value.string_value = const_cast<char*>("gtest");
    ASSERT_MOLRS_OK(molc_frame_set_meta(frame, "author", &value));

    MolcMetaValue out{};
    ASSERT_MOLRS_OK(molc_frame_get_meta(frame, "author", &out));
    EXPECT_EQ(out.dtype, MOLC_META_TYPE_STRING);
    ASSERT_NE(out.string_value, nullptr);
    EXPECT_STREQ(out.string_value, "gtest");
    molc_free_string(out.string_value);

    // missing key
    MolcMetaValue missing{};
    EXPECT_NE(molc_frame_get_meta(frame, "nope", &missing), MOLC_STATUS_OK);

    ASSERT_MOLRS_OK(molc_frame_drop(frame));
}

TEST_F(MolcTest, FrameMetadataOrder) {
    MolcFrameHandle frame{};
    ASSERT_MOLRS_OK(molc_frame_new(&frame));

    // Not alphabetical: a reintroduced sort must fail this case.
    const char* keys[] = {"zeta", "alpha", "mu"};
    for (size_t i = 0; i < 3; ++i) {
        MolcMetaValue value{};
        value.dtype = MOLC_META_TYPE_STRING;
        value.string_value = const_cast<char*>(keys[i]);
        ASSERT_MOLRS_OK(molc_frame_set_meta(frame, keys[i], &value));
    }

    uintptr_t count = 0;
    ASSERT_MOLRS_OK(molc_frame_n_meta(frame, &count));
    EXPECT_EQ(count, 3u);

    for (uintptr_t i = 0; i < 3; ++i) {
        char* out = nullptr;
        ASSERT_MOLRS_OK(molc_frame_meta_key(frame, i, &out));
        ASSERT_NE(out, nullptr);
        EXPECT_STREQ(out, keys[i]);
        molc_free_string(out);
    }

    char* out_of_range = nullptr;
    EXPECT_NE(molc_frame_meta_key(frame, 3, &out_of_range), MOLC_STATUS_OK);

    ASSERT_MOLRS_OK(molc_frame_drop(frame));
}

// ===== Block Insert & Read ================================================

TEST_F(MolcTest, BlockInsertAndRead) {
    MolcFrameHandle frame{};
    ASSERT_MOLRS_OK(molc_frame_new(&frame));

    uint32_t atoms_id = intern("gt_atoms");
    uint32_t pos_id   = intern("gt_positions");

    ASSERT_MOLRS_OK(molc_frame_set_block(frame, atoms_id));

    MolcBlockHandle block{};
    ASSERT_MOLRS_OK(molc_frame_get_block(frame, atoms_id, &block));

    // Insert 3x3 F column (F is f64).
    F data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t shape[2] = {3, 3};
    ASSERT_MOLRS_OK(molc_block_set(&block, pos_id, MOLC_D_TYPE_FLOAT, data, shape, 2));

    // nrows
    size_t nrows = 0;
    ASSERT_MOLRS_OK(molc_block_n_rows(block, &nrows));
    EXPECT_EQ(nrows, 3u);

    // ncols
    size_t ncols = 0;
    ASSERT_MOLRS_OK(molc_block_n_columns(block, &ncols));
    EXPECT_EQ(ncols, 1u);

    // dtype
    MolcDType dtype{};
    ASSERT_MOLRS_OK(molc_block_column_dtype(block, pos_id, &dtype));
    EXPECT_EQ(dtype, MOLC_D_TYPE_FLOAT);

    // shape query
    size_t col_shape[4] = {};
    size_t ndim = 4;
    ASSERT_MOLRS_OK(molc_block_column_shape(block, pos_id, col_shape, &ndim));
    EXPECT_EQ(ndim, 2u);
    EXPECT_EQ(col_shape[0], 3u);
    EXPECT_EQ(col_shape[1], 3u);

    // zero-copy read
    const uint8_t* bytes = nullptr;
    size_t len = 0;
    MolcDType got = MOLC_D_TYPE_STRING;
    ASSERT_MOLRS_OK(molc_block_get(block, pos_id, &bytes, &len, &got));
    EXPECT_EQ(got, MOLC_D_TYPE_FLOAT);
    ASSERT_EQ(len, 9u);
    const F* ptr = reinterpret_cast<const F*>(bytes);
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], data[i]);
    }

    // copy read. buf_bytes is a byte capacity.
    F buf[9] = {};
    ASSERT_MOLRS_OK(molc_block_copy(
        block, pos_id, reinterpret_cast<uint8_t*>(buf), sizeof(buf)));
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(buf[i], data[i]);
    }

    ASSERT_MOLRS_OK(molc_frame_drop(frame));
}

TEST_F(MolcTest, BlockInsertMultipleTypes) {
    MolcFrameHandle frame{};
    ASSERT_MOLRS_OK(molc_frame_new(&frame));

    uint32_t blk_id  = intern("gt_multi");
    uint32_t f_id    = intern("gt_float");
    uint32_t i_id    = intern("gt_int");
    uint32_t u_id    = intern("gt_uint");

    ASSERT_MOLRS_OK(molc_frame_set_block(frame, blk_id));

    MolcBlockHandle block{};
    ASSERT_MOLRS_OK(molc_frame_get_block(frame, blk_id, &block));

    // F column (f64)
    F f_data[3] = {-1.5f, 2.7f, 3.14f};
    size_t shape1[1] = {3};
    ASSERT_MOLRS_OK(molc_block_set(&block, f_id, MOLC_D_TYPE_FLOAT, f_data, shape1, 1));

    // I column (int32_t by default)
    int32_t i_data[3] = {100, 200, 300};
    ASSERT_MOLRS_OK(molc_block_set(&block, i_id, MOLC_D_TYPE_INT, i_data, shape1, 1));

    // U column (uint64_t / Idx)
    uint64_t u_data[3] = {1, 2, 3};
    ASSERT_MOLRS_OK(molc_block_set(&block, u_id, MOLC_D_TYPE_UINT, u_data, shape1, 1));

    // verify ncols = 3
    size_t ncols = 0;
    ASSERT_MOLRS_OK(molc_block_n_columns(block, &ncols));
    EXPECT_EQ(ncols, 3u);

    // verify dtypes
    MolcDType dt{};
    ASSERT_MOLRS_OK(molc_block_column_dtype(block, f_id, &dt));
    EXPECT_EQ(dt, MOLC_D_TYPE_FLOAT);

    ASSERT_MOLRS_OK(molc_block_column_dtype(block, i_id, &dt));
    EXPECT_EQ(dt, MOLC_D_TYPE_INT);

    ASSERT_MOLRS_OK(molc_block_column_dtype(block, u_id, &dt));
    EXPECT_EQ(dt, MOLC_D_TYPE_UINT);

    // zero-copy read F
    const uint8_t* raw = nullptr;
    size_t n = 0;
    MolcDType got = MOLC_D_TYPE_STRING;
    ASSERT_MOLRS_OK(molc_block_get(block, f_id, &raw, &n, &got));
    EXPECT_EQ(got, MOLC_D_TYPE_FLOAT);
    ASSERT_EQ(n, 3u);
    EXPECT_FLOAT_EQ(reinterpret_cast<const F*>(raw)[2], 3.14f);

    // zero-copy read I
    ASSERT_MOLRS_OK(molc_block_get(block, i_id, &raw, &n, &got));
    EXPECT_EQ(got, MOLC_D_TYPE_INT);
    ASSERT_EQ(n, 3u);
    EXPECT_EQ(reinterpret_cast<const int32_t*>(raw)[1], 200);

    // a missing column is not a dtype mismatch
    uint32_t missing = intern("gt_missing_col");
    EXPECT_EQ(molc_block_get(block, missing, &raw, &n, &got), MOLC_STATUS_KEY_NOT_FOUND);

    ASSERT_MOLRS_OK(molc_frame_drop(frame));
}

// molc_block_set takes every fixed-width dtype through its dtype argument,
// refuses a string, and refuses a bool byte other than 0 or 1.
TEST_F(MolcTest, BlockSetDtypeArgument) {
    MolcFrameHandle frame{};
    ASSERT_MOLRS_OK(molc_frame_new(&frame));

    uint32_t blk_id = intern("gt_set_dtype");
    uint32_t l_id   = intern("gt_i64");
    uint32_t b_id   = intern("gt_flag");
    uint32_t s_id   = intern("gt_str");

    ASSERT_MOLRS_OK(molc_frame_set_block(frame, blk_id));
    MolcBlockHandle block{};
    ASSERT_MOLRS_OK(molc_frame_get_block(frame, blk_id, &block));

    size_t shape[1] = {2};
    int64_t l_data[2] = {-7, 1LL << 40};
    ASSERT_MOLRS_OK(molc_block_set(&block, l_id, MOLC_D_TYPE_I64, l_data, shape, 1));
    const uint8_t* raw = nullptr;
    size_t n = 0;
    MolcDType got = MOLC_D_TYPE_STRING;
    ASSERT_MOLRS_OK(molc_block_get(block, l_id, &raw, &n, &got));
    EXPECT_EQ(got, MOLC_D_TYPE_I64);
    ASSERT_EQ(n, 2u);
    EXPECT_EQ(reinterpret_cast<const int64_t*>(raw)[1], 1LL << 40);

    bool flags[2] = {true, false};
    ASSERT_MOLRS_OK(molc_block_set(&block, b_id, MOLC_D_TYPE_BOOL, flags, shape, 1));
    ASSERT_MOLRS_OK(molc_block_column_dtype(block, b_id, &got));
    EXPECT_EQ(got, MOLC_D_TYPE_BOOL);

    uint8_t bad_flags[2] = {1, 2};
    EXPECT_EQ(molc_block_set(&block, b_id, MOLC_D_TYPE_BOOL, bad_flags, shape, 1),
              MOLC_STATUS_INVALID_ARGUMENT);

    const char* words[2] = {"a", "b"};
    EXPECT_EQ(molc_block_set(&block, s_id, MOLC_D_TYPE_STRING, words, shape, 1),
              MOLC_STATUS_TYPE_MISMATCH);

    ASSERT_MOLRS_OK(molc_frame_drop(frame));
}

// ===== Block Mutable Pointer ==============================================

TEST_F(MolcTest, BlockMutablePointer) {
    MolcFrameHandle frame{};
    ASSERT_MOLRS_OK(molc_frame_new(&frame));

    uint32_t blk_id = intern("gt_mut_blk");
    uint32_t x_id   = intern("gt_x");

    ASSERT_MOLRS_OK(molc_frame_set_block(frame, blk_id));

    MolcBlockHandle block{};
    ASSERT_MOLRS_OK(molc_frame_get_block(frame, blk_id, &block));

    F data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t shape[1] = {4};
    ASSERT_MOLRS_OK(molc_block_set(&block, x_id, MOLC_D_TYPE_FLOAT, data, shape, 1));

    // get mutable pointer
    uint8_t* raw = nullptr;
    size_t len = 0;
    MolcDType got = MOLC_D_TYPE_STRING;
    ASSERT_MOLRS_OK(molc_block_get_mut(&block, x_id, &raw, &len, &got));
    EXPECT_EQ(got, MOLC_D_TYPE_FLOAT);
    ASSERT_EQ(len, 4u);
    F* ptr = reinterpret_cast<F*>(raw);

    // modify in-place
    for (size_t i = 0; i < len; ++i) {
        ptr[i] *= 10.0f;
    }

    // verify via copy
    F buf[4] = {};
    ASSERT_MOLRS_OK(molc_block_copy(
        block, x_id, reinterpret_cast<uint8_t*>(buf), sizeof(buf)));
    EXPECT_FLOAT_EQ(buf[0], 10.0f);
    EXPECT_FLOAT_EQ(buf[1], 20.0f);
    EXPECT_FLOAT_EQ(buf[2], 30.0f);
    EXPECT_FLOAT_EQ(buf[3], 40.0f);

    ASSERT_MOLRS_OK(molc_frame_drop(frame));
}

// ===== Box =============================================================

TEST_F(MolcTest, BoxCube) {
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolcBoxHandle sb{};
    F edge[3] = {10, 10, 10};
    ASSERT_MOLRS_OK(molc_box_new(MOLC_CELL_KIND_LENGTHS, edge, origin, pbc, &sb));

    F vol = 0;
    ASSERT_MOLRS_OK(molc_box_volume(sb, &vol));
    EXPECT_NEAR(vol, 1000.0, 1e-3);

    F lengths[3] = {};
    ASSERT_MOLRS_OK(molc_box_lengths(sb, lengths));
    EXPECT_NEAR(lengths[0], 10.0, 1e-6);
    EXPECT_NEAR(lengths[1], 10.0, 1e-6);
    EXPECT_NEAR(lengths[2], 10.0, 1e-6);

    F h[9] = {};
    ASSERT_MOLRS_OK(molc_box_h(sb, h));
    EXPECT_NEAR(h[0], 10.0, 1e-6);  // h[0][0]
    EXPECT_NEAR(h[4], 10.0, 1e-6);  // h[1][1]
    EXPECT_NEAR(h[8], 10.0, 1e-6);  // h[2][2]
    EXPECT_NEAR(h[1], 0.0, 1e-10);  // off-diagonal

    F tilts[3] = {};
    ASSERT_MOLRS_OK(molc_box_tilts(sb, tilts));
    EXPECT_NEAR(tilts[0], 0.0, 1e-10);
    EXPECT_NEAR(tilts[1], 0.0, 1e-10);
    EXPECT_NEAR(tilts[2], 0.0, 1e-10);

    bool pbc_out[3] = {};
    ASSERT_MOLRS_OK(molc_box_pbc(sb, pbc_out));
    EXPECT_TRUE(pbc_out[0]);
    EXPECT_TRUE(pbc_out[1]);
    EXPECT_TRUE(pbc_out[2]);

    ASSERT_MOLRS_OK(molc_box_drop(sb));
    EXPECT_NE(molc_box_drop(sb), MOLC_STATUS_OK);  // double drop
}

TEST_F(MolcTest, BoxOrtho) {
    F lens[3] = {2, 3, 4};
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolcBoxHandle sb{};
    ASSERT_MOLRS_OK(molc_box_new(MOLC_CELL_KIND_LENGTHS, lens, origin, pbc, &sb));

    F vol = 0;
    ASSERT_MOLRS_OK(molc_box_volume(sb, &vol));
    EXPECT_NEAR(vol, 24.0, 1e-3);

    ASSERT_MOLRS_OK(molc_box_drop(sb));
}

TEST_F(MolcTest, BoxWrap) {
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolcBoxHandle sb{};
    F edge[3] = {10, 10, 10};
    ASSERT_MOLRS_OK(molc_box_new(MOLC_CELL_KIND_LENGTHS, edge, origin, pbc, &sb));

    // (11, -1, 21) -> should wrap to (1, 9, 1)
    F xyz_in[6]  = {11, -1, 21, 0.5, 0.5, 0.5};
    F xyz_out[6] = {};
    ASSERT_MOLRS_OK(molc_box_wrap(sb, xyz_in, xyz_out, 2));

    EXPECT_NEAR(xyz_out[0], 1.0, 1e-4);
    EXPECT_NEAR(xyz_out[1], 9.0, 1e-4);
    EXPECT_NEAR(xyz_out[2], 1.0, 1e-4);

    ASSERT_MOLRS_OK(molc_box_drop(sb));
}

TEST_F(MolcTest, BoxDelta) {
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolcBoxHandle sb{};
    F edge[3] = {10, 10, 10};
    ASSERT_MOLRS_OK(molc_box_new(MOLC_CELL_KIND_LENGTHS, edge, origin, pbc, &sb));

    F r1[3] = {0.5, 0, 0};
    F r2[3] = {9.5, 0, 0};
    F dr[3] = {};
    ASSERT_MOLRS_OK(molc_box_delta(sb, r1, r2, true, dr, 1));

    // shortest path across periodic boundary: -1.0 in x
    EXPECT_NEAR(dr[0], -1.0, 1e-4);
    EXPECT_NEAR(dr[1],  0.0, 1e-4);
    EXPECT_NEAR(dr[2],  0.0, 1e-4);

    // without the minimum image: the plain difference
    ASSERT_MOLRS_OK(molc_box_delta(sb, r1, r2, false, dr, 1));
    EXPECT_NEAR(dr[0], 9.0, 1e-4);

    ASSERT_MOLRS_OK(molc_box_drop(sb));
}

TEST_F(MolcTest, BoxTriclinic) {
    // upper-triangular cell matrix
    F h9[9] = {
        2, 1, 2,
        0, 4, 3,
        0, 0, 5,
    };
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolcBoxHandle sb{};
    ASSERT_MOLRS_OK(molc_box_new(MOLC_CELL_KIND_MATRIX, h9, origin, pbc, &sb));

    F tilts[3] = {};
    ASSERT_MOLRS_OK(molc_box_tilts(sb, tilts));
    EXPECT_NEAR(tilts[0], 1.0, 1e-6);  // xy
    EXPECT_NEAR(tilts[1], 2.0, 1e-6);  // xz
    EXPECT_NEAR(tilts[2], 3.0, 1e-6);  // yz

    ASSERT_MOLRS_OK(molc_box_drop(sb));
}

// The cell kinds a Box is built from, beyond lengths and a matrix.
TEST_F(MolcTest, BoxCellKinds) {
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};

    // LAMMPS form: diagonal sizes then tilts xy, xz, yz.
    F lt[6] = {2, 4, 5, 1, 2, 3};
    MolcBoxHandle tilted{};
    ASSERT_MOLRS_OK(molc_box_new(MOLC_CELL_KIND_LENGTHS_TILTS, lt, origin, pbc, &tilted));
    F tilts[3] = {};
    ASSERT_MOLRS_OK(molc_box_tilts(tilted, tilts));
    EXPECT_NEAR(tilts[0], 1.0, 1e-12);
    EXPECT_NEAR(tilts[2], 3.0, 1e-12);
    ASSERT_MOLRS_OK(molc_box_drop(tilted));

    // lengths and right angles: an orthorhombic cell.
    F la[6] = {2, 3, 4, 90, 90, 90};
    MolcBoxHandle ortho{};
    ASSERT_MOLRS_OK(molc_box_new(MOLC_CELL_KIND_LENGTHS_ANGLES, la, origin, pbc, &ortho));
    F vol = 0;
    ASSERT_MOLRS_OK(molc_box_volume(ortho, &vol));
    EXPECT_NEAR(vol, 24.0, 1e-9);
    ASSERT_MOLRS_OK(molc_box_drop(ortho));

    // no cell: no values are read.
    bool open[3] = {false, false, false};
    MolcBoxHandle none{};
    ASSERT_MOLRS_OK(molc_box_new(MOLC_CELL_KIND_UNDEFINED, nullptr, origin, open, &none));
    ASSERT_MOLRS_OK(molc_box_drop(none));

    // a kind that reads values refuses a null cell; a bad length is refused.
    MolcBoxHandle sb{};
    EXPECT_EQ(molc_box_new(MOLC_CELL_KIND_LENGTHS, nullptr, origin, pbc, &sb),
              MOLC_STATUS_NULL_POINTER);
    F negative[3] = {1, -1, 1};
    EXPECT_EQ(molc_box_new(MOLC_CELL_KIND_LENGTHS, negative, origin, pbc, &sb),
              MOLC_STATUS_INVALID_ARGUMENT);
    F singular[9] = {1, 0, 0, 0, 0, 0, 0, 0, 1};
    EXPECT_EQ(molc_box_new(MOLC_CELL_KIND_MATRIX, singular, origin, pbc, &sb),
              MOLC_STATUS_SINGULAR_CELL);
}

// ===== Frame <-> Box =====================================================

TEST_F(MolcTest, FrameBoxAssociation) {
    MolcFrameHandle frame{};
    ASSERT_MOLRS_OK(molc_frame_new(&frame));

    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolcBoxHandle sb{};
    F edge[3] = {5, 5, 5};
    ASSERT_MOLRS_OK(molc_box_new(MOLC_CELL_KIND_LENGTHS, edge, origin, pbc, &sb));

    ASSERT_MOLRS_OK(molc_frame_set_box(frame, sb));

    MolcBoxHandle sb2{};
    ASSERT_MOLRS_OK(molc_frame_get_box(frame, &sb2));

    F vol = 0;
    ASSERT_MOLRS_OK(molc_box_volume(sb2, &vol));
    EXPECT_NEAR(vol, 125.0, 1e-3);

    // delete and verify absence
    ASSERT_MOLRS_OK(molc_frame_del_box(frame));
    MolcBoxHandle sb3{};
    EXPECT_NE(molc_frame_get_box(frame, &sb3), MOLC_STATUS_OK);

    molc_box_drop(sb);
    molc_box_drop(sb2);
    ASSERT_MOLRS_OK(molc_frame_drop(frame));
}

// ===== ForceField =========================================================

TEST_F(MolcTest, ForceFieldLifecycle) {
    MolcForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molc_forcefield_new("gtest_ff", &ff));

    ASSERT_MOLRS_OK(molc_forcefield_def_style(ff, "bond", "harmonic", nullptr, nullptr, 0));
    ASSERT_MOLRS_OK(molc_forcefield_def_style(ff, "angle", "harmonic", nullptr, nullptr, 0));
    ASSERT_MOLRS_OK(molc_forcefield_def_style(ff, "atom", "full", nullptr, nullptr, 0));

    size_t count = 0;
    ASSERT_MOLRS_OK(molc_forcefield_n_styles(ff, &count));
    EXPECT_EQ(count, 3u);

    // query style name
    char* cat = nullptr;
    char* name = nullptr;
    ASSERT_MOLRS_OK(molc_forcefield_style_name(ff, 0, &cat, &name));
    ASSERT_NE(cat, nullptr);
    ASSERT_NE(name, nullptr);
    molc_free_string(cat);
    molc_free_string(name);

    ASSERT_MOLRS_OK(molc_forcefield_drop(ff));
    EXPECT_NE(molc_forcefield_drop(ff), MOLC_STATUS_OK);  // double drop
}

TEST_F(MolcTest, ForceFieldPairStyle) {
    MolcForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molc_forcefield_new("gtest_pair", &ff));

    const char* style_pk[] = {"cutoff"};
    double style_pv[] = {10.0};
    ASSERT_MOLRS_OK(molc_forcefield_def_style(ff, "pair", "lj/cut", style_pk, style_pv, 1));

    const char* type_pk[] = {"epsilon", "sigma"};
    double type_pv[] = {0.5, 3.4};
    const char* ar[] = {"Ar"};
    const char* ar_kr[] = {"Ar", "Kr"};
    ASSERT_MOLRS_OK(molc_forcefield_def_type(ff, "pair", "lj/cut", "Ar", ar, 1, type_pk, type_pv, 2));
    ASSERT_MOLRS_OK(
        molc_forcefield_def_type(ff, "pair", "lj/cut", "Ar-Kr", ar_kr, 2, type_pk, type_pv, 2));

    size_t count = 0;
    ASSERT_MOLRS_OK(molc_forcefield_n_styles(ff, &count));
    EXPECT_EQ(count, 1u);

    ASSERT_MOLRS_OK(molc_forcefield_drop(ff));
}

TEST_F(MolcTest, ForceFieldDefStyleDefTypeAreOk) {
    MolcForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molc_forcefield_new("gtest_primitives", &ff));

    ASSERT_MOLRS_OK(molc_forcefield_def_style(ff, "bond", "harmonic", nullptr, nullptr, 0));

    const char* pk[] = {"k0", "r0"};
    double pv[] = {300.0, 1.4};
    const char* ct_oh[] = {"CT", "OH"};
    ASSERT_MOLRS_OK(molc_forcefield_def_type(ff, "bond", "harmonic", "CT-OH", ct_oh, 2, pk, pv, 2));

    // The name is opaque: MMFF's `0_1_5` is defined on the endpoints given.
    const char* endpoints[] = {"1", "5"};
    ASSERT_MOLRS_OK(
        molc_forcefield_def_type(ff, "bond", "harmonic", "0_1_5", endpoints, 2, pk, pv, 2));

    ASSERT_MOLRS_OK(molc_forcefield_drop(ff));
}

TEST_F(MolcTest, ForceFieldDefStyleUnknownCategoryIsInvalidArgument) {
    MolcForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molc_forcefield_new("gtest_bad_category", &ff));

    EXPECT_EQ(molc_forcefield_def_style(ff, "kspace", "pme", nullptr, nullptr, 0),
              MOLC_STATUS_INVALID_ARGUMENT);

    ASSERT_MOLRS_OK(molc_forcefield_drop(ff));
}

TEST_F(MolcTest, ForceFieldDefTypeOnMissingStyleIsInvalidArgument) {
    MolcForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molc_forcefield_new("gtest_missing_style", &ff));

    const char* pk[] = {"k0", "r0"};
    double pv[] = {300.0, 1.4};
    const char* ct_oh[] = {"CT", "OH"};
    // No style is created implicitly.
    EXPECT_EQ(molc_forcefield_def_type(ff, "bond", "harmonic", "CT-OH", ct_oh, 2, pk, pv, 2),
              MOLC_STATUS_INVALID_ARGUMENT);

    ASSERT_MOLRS_OK(molc_forcefield_drop(ff));
}

TEST_F(MolcTest, ForceFieldDefTypeWrongEndpointCountIsInvalidArgument) {
    MolcForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molc_forcefield_new("gtest_malformed", &ff));
    ASSERT_MOLRS_OK(molc_forcefield_def_style(ff, "bond", "harmonic", nullptr, nullptr, 0));

    const char* pk[] = {"k0", "r0"};
    double pv[] = {300.0, 1.4};
    const char* ct[] = {"CT"};
    // One endpoint on a bond style: an error status, never a panic.
    EXPECT_EQ(molc_forcefield_def_type(ff, "bond", "harmonic", "CT-CT", ct, 1, pk, pv, 2),
              MOLC_STATUS_INVALID_ARGUMENT);

    ASSERT_MOLRS_OK(molc_forcefield_drop(ff));
}

TEST_F(MolcTest, ForceFieldJsonRoundtrip) {
    MolcForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molc_forcefield_new("gtest_json", &ff));

    const char* spk[] = {"cutoff"};
    double spv[] = {12.0};
    ASSERT_MOLRS_OK(molc_forcefield_def_style(ff, "pair", "lj/cut", spk, spv, 1));

    const char* tpk[] = {"epsilon", "sigma"};
    double tpv[] = {1.0, 3.4};
    const char* ar[] = {"Ar"};
    ASSERT_MOLRS_OK(molc_forcefield_def_type(ff, "pair", "lj/cut", "Ar", ar, 1, tpk, tpv, 2));

    // serialize: the core forcefield section (ForceFieldSection::from_forcefield) as JSON
    char* json = nullptr;
    size_t json_len = 0;
    ASSERT_MOLRS_OK(molc_forcefield_to_json(ff, &json, &json_len));
    ASSERT_NE(json, nullptr);
    EXPECT_GT(json_len, 0u);
    EXPECT_NE(std::string(json).find("\"document\""), std::string::npos);
    EXPECT_NE(std::string(json).find("\"tables\""), std::string::npos);

    // deserialize
    MolcForceFieldHandle ff2{};
    ASSERT_MOLRS_OK(molc_forcefield_from_json(json, &ff2));

    size_t count = 0;
    ASSERT_MOLRS_OK(molc_forcefield_n_styles(ff2, &count));
    EXPECT_EQ(count, 1u);

    molc_free_string(json);
    ASSERT_MOLRS_OK(molc_forcefield_drop(ff));
    ASSERT_MOLRS_OK(molc_forcefield_drop(ff2));
}

// ===== Error Handling =====================================================

TEST_F(MolcTest, NullPointerErrors) {
    EXPECT_EQ(molc_frame_new(nullptr), MOLC_STATUS_NULL_POINTER);
    EXPECT_EQ(molc_intern_key(nullptr, nullptr), MOLC_STATUS_NULL_POINTER);
    EXPECT_EQ(molc_box_new(MOLC_CELL_KIND_LENGTHS, nullptr, nullptr, nullptr, nullptr),
              MOLC_STATUS_NULL_POINTER);
}

TEST_F(MolcTest, InvalidHandleErrors) {
    MolcFrameHandle bad_frame{999, 999};
    EXPECT_NE(molc_frame_drop(bad_frame), MOLC_STATUS_OK);

    MolcBoxHandle bad_sb{999, 999};
    EXPECT_NE(molc_box_drop(bad_sb), MOLC_STATUS_OK);

    MolcForceFieldHandle bad_ff{999, 999};
    EXPECT_NE(molc_forcefield_drop(bad_ff), MOLC_STATUS_OK);
}

TEST_F(MolcTest, BlockCopyBufferTooSmall) {
    MolcFrameHandle frame{};
    ASSERT_MOLRS_OK(molc_frame_new(&frame));

    uint32_t blk_id = intern("gt_small_buf_blk");
    uint32_t col_id = intern("gt_small_buf_col");

    ASSERT_MOLRS_OK(molc_frame_set_block(frame, blk_id));
    MolcBlockHandle block{};
    ASSERT_MOLRS_OK(molc_frame_get_block(frame, blk_id, &block));

    F data[5] = {1, 2, 3, 4, 5};
    size_t shape[1] = {5};
    ASSERT_MOLRS_OK(molc_block_set(&block, col_id, MOLC_D_TYPE_FLOAT, data, shape, 1));

    // buffer too small: two elements, counted in bytes
    F small_buf[2] = {};
    MolcStatus s = molc_block_copy(
        block, col_id, reinterpret_cast<uint8_t*>(small_buf), sizeof(small_buf));
    EXPECT_EQ(s, MOLC_STATUS_INVALID_ARGUMENT);

    ASSERT_MOLRS_OK(molc_frame_drop(frame));
}

TEST(Abi, MolcVersionIsANonEmptyDottedString) {
    const char* v = molc_version();
    ASSERT_NE(v, nullptr);
    std::string s(v);
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find('.'), std::string::npos);
}

TEST(Schema, JsonIsOwnedNonEmptyAndFreeable) {
    char* json = molc_schema_document();
    ASSERT_NE(json, nullptr);
    std::string s(json);
    molc_free_string(json);
    EXPECT_NE(s.find("\"id\""), std::string::npos);
    EXPECT_NE(s.find("\"columns\""), std::string::npos);
    EXPECT_NE(s.find("\"blocks\""), std::string::npos);
}

TEST(Schema, CountsAreNonZero) {
    EXPECT_GT(molc_schema_n_columns(), 0u);
    EXPECT_GT(molc_schema_n_blocks(), 0u);
}

TEST(Schema, IdentifiersAreUnsignedAndTypeIsAString) {
    // The two decisions the vocabulary is built on, asserted from C: every
    // identifier is unsigned, and `type` is a label rather than an ordinal.
    for (const char* key : {"id", "mol_id", "res_id", "type_id",
                            "atomi", "atomj", "atomk", "atoml"}) {
        const char* dt = molc_schema_column_dtype(key);
        ASSERT_NE(dt, nullptr) << key;
        EXPECT_STREQ(dt, "uint") << key;
    }
    EXPECT_STREQ(molc_schema_column_dtype("type"), "string");
    EXPECT_STREQ(molc_schema_column_dtype("x"), "float");
}

TEST(Schema, UnconstrainedKeyIsNullNotAnError) {
    // A NULL dtype means "no declared dtype", not "invalid key": unspecified
    // keys are the documented extension point.
    EXPECT_EQ(molc_schema_column_dtype("some_local_column"), nullptr);
    EXPECT_EQ(molc_schema_column_dtype(nullptr), nullptr);
}

TEST(Schema, KnownBlocksAreReportedAndTheBlockSetStaysOpen) {
    EXPECT_TRUE(molc_schema_has_block("atoms"));
    EXPECT_TRUE(molc_schema_has_block("bonds"));
    // Not in the vocabulary — legal, just unnamed.
    EXPECT_FALSE(molc_schema_has_block("my_relation"));
    EXPECT_FALSE(molc_schema_has_block(nullptr));
}

// ===== Regions ==============================================================

TEST_F(MolcTest, RegionCompositionIsDeclaredAndComposes) {
    // and / or / not are declared in molc.h, so a C++ caller links them.
    const F origin[3] = {0, 0, 0};
    MolcRegionHandle outer, inner, hole, shell, both;
    ASSERT_MOLRS_OK(molc_region_sphere(origin, 3.0, &outer));
    ASSERT_MOLRS_OK(molc_region_sphere(origin, 2.0, &inner));
    ASSERT_MOLRS_OK(molc_region_not(inner, &hole));
    ASSERT_MOLRS_OK(molc_region_and(outer, hole, &shell));
    ASSERT_MOLRS_OK(molc_region_or(inner, shell, &both));

    const F points[6] = {2.5, 0, 0, 1.0, 0, 0};
    bool in_shell[2] = {false, false};
    bool in_both[2] = {false, false};
    ASSERT_MOLRS_OK(molc_region_contains(shell, points, 2, in_shell));
    ASSERT_MOLRS_OK(molc_region_contains(both, points, 2, in_both));
    EXPECT_TRUE(in_shell[0]);
    EXPECT_FALSE(in_shell[1]);
    EXPECT_TRUE(in_both[0]);
    EXPECT_TRUE(in_both[1]);

    for (MolcRegionHandle h : {both, shell, hole, inner, outer}) {
        EXPECT_MOLRS_OK(molc_region_drop(h));
    }
}

// A parallelepiped region takes its cell the way a Box does.
TEST_F(MolcTest, RegionParallelepipedTakesACell) {
    F edge[3] = {2, 2, 2};
    F origin[3] = {0, 0, 0};
    MolcRegionHandle cube{};
    ASSERT_MOLRS_OK(molc_region_parallelepiped(MOLC_CELL_KIND_LENGTHS, edge, origin, &cube));
    F points[6] = {1, 1, 1, 3, 1, 1};
    bool inside[2] = {};
    ASSERT_MOLRS_OK(molc_region_contains(cube, points, 2, inside));
    EXPECT_TRUE(inside[0]);
    EXPECT_FALSE(inside[1]);
    ASSERT_MOLRS_OK(molc_region_drop(cube));

    MolcRegionHandle none{};
    EXPECT_EQ(molc_region_parallelepiped(MOLC_CELL_KIND_UNDEFINED, nullptr, origin, &none),
              MOLC_STATUS_INVALID_ARGUMENT);
}

TEST_F(MolcTest, ShutdownDropsRegions) {
    // molc_shutdown resets the whole handle registry: a region obtained
    // before it is stale afterwards, like every other handle.
    const F origin[3] = {0, 0, 0};
    MolcRegionHandle sphere;
    ASSERT_MOLRS_OK(molc_region_sphere(origin, 1.0, &sphere));
    molc_shutdown();
    molc_init();
    F d = 0.0;
    EXPECT_EQ(molc_region_distance(sphere, origin, 1, &d), MOLC_STATUS_INVALID_REGION_HANDLE);
}
