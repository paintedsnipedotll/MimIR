#include <cassert>

#include <iostream>

#include <mim/lam.h>
#include <mim/pass.h>

#include "mim/plug/aie2p/autogen.h"

namespace mim::plug::aie2p {

struct LowerAIE2PPass final : Pass {
    Lam* llvm_get_coreid_ = nullptr; // external lam: [] -> Cn I32
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

            if (!llvm_get_coreid_) {
                assert(app->arg() && "get_coreid: missing unit arg");
                assert(app->arg()->type() && "get_coreid: missing unit arg type");

                auto dom = app->arg()->type();
                auto ret = app->type();

                llvm_get_coreid_ = w.mut_lam(dom, ret);
                llvm_get_coreid_->set("llvm.aie2p.get.coreid");
                llvm_get_coreid_->externalize();
                std::cerr << "[aie2p] created external lam llvm.aie2p.get.coreid\n";
            }

            return w.raw_app(app->type(), llvm_get_coreid_, app->arg());
        }

        return Pass::rewrite(def);
    }
};

void register_stages(Flags2Stages& stages) {
    std::cerr << "[aie2p] register_stages\n";
    Stage::hook<lower_pass, LowerAIE2PPass>(stages);
}

} // namespace mim::plug::aie2p
