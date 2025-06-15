#include <catch2/catch.hpp>
#include "molpy_core/Frame.hpp"

using namespace molpy;

TEST_CASE("Frame basic") {
    Frame f;
    Table t;
    t["col1"] = Array({1.0,2.0,3.0});
    f.add_table("atoms", t);
    REQUIRE(f.size("atoms") == 3);
    auto& r = f.table("atoms")["col1"];
    REQUIRE(r(0) == Approx(1.0));
}

TEST_CASE("Frame concat") {
    Table t1; t1["a"] = Array({1.0});
    Table t2; t2["a"] = Array({2.0});
    Frame f1({{"x",t1}}), f2({{"x",t2}});
    Frame f3 = Frame::concat(f1, f2);
    REQUIRE(f3.size("x") == 2);
    REQUIRE(f3.table("x") == Approx(2.0));
}
