#pragma once

#include <mim/phase.h>

namespace mim::plug::aie2p::phase {

/// Lowers %aie2p.get_coreid to the CPS form expected by the LLVM backend:
/// llvm.aie2p.get.coreid : Cn [[], Cn I32], wrapped with direct.cps2ds_dep.
class LowerAIE2P : public RWPhase {
public:
    LowerAIE2P(World& world, flags_t annex)
        : RWPhase(world, annex) {}

    const Def* rewrite_imm_App(const App*) final;

private:
    const Def* llvm_get_coreid_wrapped_ = nullptr;
    const Def* llvm_clb_wrapped_ = nullptr;
};

} // namespace mim::plug::aie2p::phase
