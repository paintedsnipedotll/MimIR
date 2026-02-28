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
    if (!cached_wrapped) {
        auto cn_ret = new_w.cn(ret);

        if (auto sigma = dom->isa<Sigma>(); sigma && sigma->num_ops() >= 2) {
            // Multi-arg intrinsic (e.g. SRS): create flat LLVM-facing lambda + forwarding lambda.
            // op_cps2ds_dep assumes a 2-element domain, so we can't pass the flat lambda directly.

            // 1. Flat LLVM-facing lambda (external, no body)
            DefVec flat_doms;
            for (auto op : sigma->ops()) flat_doms.push_back(op);
            flat_doms.push_back(cn_ret);
            auto flat_lam = new_w.mut_con(flat_doms)->set(llvm_name);

            // 2. Forwarding lambda (2-element domain for cps2ds_dep compatibility)
            auto fwd_lam   = new_w.mut_con({dom, cn_ret})->set("fwd");
            auto sigma_var = fwd_lam->var(2, 0);
            auto cont_var  = fwd_lam->var(2, 1);
            DefVec call_args;
            for (size_t i = 0; i < sigma->num_ops(); ++i)
                call_args.push_back(sigma_var->proj(sigma->num_ops(), i));
            call_args.push_back(cont_var);
            fwd_lam->app(true, flat_lam, call_args);

            cached_wrapped = direct::op_cps2ds_dep(fwd_lam);
        } else {
            // Single-arg / unit intrinsic (e.g. CLB, get_coreid): 2-element domain works directly.
            auto cps_lam = new_w.mut_con({dom, cn_ret})->set(llvm_name);
            cached_wrapped = direct::op_cps2ds_dep(cps_lam);
        }
    }
    return new_w.app(cached_wrapped, arg_rewritten);
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
