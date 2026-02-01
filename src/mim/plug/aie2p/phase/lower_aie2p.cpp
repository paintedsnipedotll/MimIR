#include "mim/plug/aie2p/phase/lower_aie2p.h"

#include <cassert>

#include <mim/lam.h>

#include "mim/plug/aie2p/autogen.h"
#include "mim/plug/direct/direct.h"

namespace mim::plug::aie2p::phase {

static const Def* lower_to_cps_intrinsic(World& new_w, const Def* arg_rewritten,
                                        const Def* dom, const Def* ret,
                                        const char* llvm_name,
                                        const Def*& cached_wrapped) {
    const Def* wrapped = nullptr;
    if (!cached_wrapped) {
        auto sigma = new_w.mut_sigma(2);
        sigma->set(0, dom);
        sigma->set(1, new_w.cn(ret));
        auto cps_lam = new_w.mut_con(sigma)->set(llvm_name);
        wrapped = direct::op_cps2ds_dep(cps_lam);
        cached_wrapped = wrapped;
    } else {
        wrapped = cached_wrapped;
    }
    return new_w.app(wrapped, arg_rewritten);
}

const Def* LowerAIE2P::rewrite_imm_App(const App* app) {
    if (is_bootstrapping()) return Rewriter::rewrite_imm_App(app);

    auto& new_w = new_world();
    auto arg_rewritten = rewrite(app->arg());
    if (!arg_rewritten) return Rewriter::rewrite_imm_App(app);

    if (Axm::isa<get_coreid>(app)) {
        assert(app->arg() && app->arg()->type() && "get_coreid: missing unit arg/type");
        auto dom = rewrite(app->arg()->type());
        auto ret = rewrite(app->type());
        if (!dom || !ret) return Rewriter::rewrite_imm_App(app);
        return lower_to_cps_intrinsic(new_w, arg_rewritten, dom, ret,
                                      "llvm.aie2p.get.coreid", llvm_get_coreid_wrapped_);
    }

    if (Axm::isa<clb>(app)) {
        assert(app->arg() && app->arg()->type() && "clb: missing arg/type");
        auto dom = rewrite(app->arg()->type());
        auto ret = rewrite(app->type());
        if (!dom || !ret) return Rewriter::rewrite_imm_App(app);
        return lower_to_cps_intrinsic(new_w, arg_rewritten, dom, ret,
                                      "llvm.aie2p.clb", llvm_clb_wrapped_);
    }

    if (Axm::isa<srs_i16_32>(app)) {
        assert(app->arg() && app->arg()->type() && "srs_i16_32: missing arg/type");
        auto dom = rewrite(app->arg()->type());
        auto ret = rewrite(app->type());
        if (!dom || !ret) return Rewriter::rewrite_imm_App(app);
        return lower_to_cps_intrinsic(new_w, arg_rewritten, dom, ret,
                                      "llvm.aie2p.I512.v32.acc64.srs", llvm_srs_i16_32_wrapped_);
    }

    return Rewriter::rewrite_imm_App(app);
}

} // namespace mim::plug::aie2p::phase
