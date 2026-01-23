#include <iostream>

#include "mim/world.h"

#include "mim/plug/aie2p/autogen.h"

namespace mim::plug::aie2p {

const Def* normalize_mul_scalar(const Def*, const Def*, const Def* arg) {
    auto [x, y] = arg->projs<2>();
    (void)x;
    return y;
}

const Def* normalize_get_coreid(const Def* type, const Def* callee, const Def* arg) {
    std::cerr << "[aie2p] normalize_get_coreid (raw_app)\n";
    auto& w = callee->world();
    return w.raw_app(type, callee, arg);
}

MIM_aie2p_NORMALIZER_IMPL

} // namespace mim::plug::aie2p
