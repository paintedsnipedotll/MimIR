#include <cassert>

#include <iostream>

#include <mim/lam.h>
#include <mim/pass.h>

#include "mim/plug/aie2p/autogen.h"
#include "mim/plug/direct/direct.h"

namespace mim::plug::aie2p {

struct LowerAIE2PPass final : Pass {
    const Def* llvm_get_coreid_wrapped_ = nullptr; // wrapped CPS function: cps2ds_dep (Cn [[], Cn I32])
    size_t calls_ = 0, hits_ = 0;

    explicit LowerAIE2PPass(World& w, flags_t annex)
        : Pass(w, annex) {
        std::cerr << "[aie2p] LowerAIE2PPass ctor\n";
    }

    ~LowerAIE2PPass() override { std::cerr << "[aie2p] dtor calls=" << calls_ << " hits=" << hits_ << "\n"; }

    bool inspect() const override { return man().curr_mut(); }

    const Def* rewrite(const Def* def) override {
        ++calls_;

        auto app = def->isa<App>();
        if (!app) return Pass::rewrite(def);

        if (Axm::isa<get_coreid>(app)) {
            ++hits_;
            std::cerr << "[aie2p] HIT %aie2p.get_coreid\n";

            auto& w = app->world();

            if (!llvm_get_coreid_wrapped_) {
                assert(app->arg() && "get_coreid: missing unit arg");
                assert(app->arg()->type() && "get_coreid: missing unit arg type");

                auto dom = app->arg()->type();  // [] (unit)
                auto ret = app->type();         // I32

                // Create CPS function type: Cn [dom, Cn ret]
                // For get_coreid: [] -> I32, this becomes Cn [[], Cn I32]
                auto sigma = w.mut_sigma(2);
                sigma->set(0, dom);
                sigma->set(1, w.cn(ret));

                // Create the CPS lambda (unset, no body - will be declared as external by LLVM backend)
                auto cps_lam = w.mut_con(sigma)->set("llvm.aie2p.get.coreid");
                // Don't call externalize() - leave it unset

                // Wrap with cps2ds to get direct-style interface
                llvm_get_coreid_wrapped_ = direct::op_cps2ds_dep(cps_lam);
                std::cerr << "[aie2p] created CPS lam llvm.aie2p.get.coreid wrapped with cps2ds\n";
            }

            return w.app(llvm_get_coreid_wrapped_, app->arg());
        }

        return Pass::rewrite(def);
    }
};

void register_stages(Flags2Stages& stages) {
    std::cerr << "[aie2p] register_stages\n";
    Stage::hook<lower_pass, LowerAIE2PPass>(stages);
}

} // namespace mim::plug::aie2p
