#include "mim/plug/aie2p/phase/lower_aie2p.h"

#include <cassert>

#include <mim/lam.h>

#include "mim/plug/aie2p/autogen.h"
#include "mim/plug/direct/direct.h"

namespace mim::plug::aie2p::phase {

const Def* LowerAIE2P::rewrite_imm_App(const App* app) {
    if (is_bootstrapping()) return Rewriter::rewrite_imm_App(app);

    if (Axm::isa<get_coreid>(app)) {
        auto& new_w = new_world();

        auto arg_rewritten = rewrite(app->arg());
        if (!arg_rewritten) return Rewriter::rewrite_imm_App(app);

        const Def* wrapped = nullptr;
        if (!llvm_get_coreid_wrapped_) {
            assert(app->arg() && app->arg()->type() && "get_coreid: missing unit arg/type");

            auto dom = rewrite(app->arg()->type());
            auto ret = rewrite(app->type());
            if (!dom || !ret) return Rewriter::rewrite_imm_App(app);

            auto sigma = new_w.mut_sigma(2);
            sigma->set(0, dom);
            sigma->set(1, new_w.cn(ret));

            auto cps_lam = new_w.mut_con(sigma)->set("llvm.aie2p.get.coreid");
            wrapped = direct::op_cps2ds_dep(cps_lam);
            llvm_get_coreid_wrapped_ = wrapped;
        } else {
            wrapped = llvm_get_coreid_wrapped_;
        }

        return new_w.app(wrapped, arg_rewritten);
    }

    return Rewriter::rewrite_imm_App(app);
}

} // namespace mim::plug::aie2p::phase
