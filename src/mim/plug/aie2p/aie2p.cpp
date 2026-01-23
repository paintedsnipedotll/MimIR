#include <mim/config.h>
#include <mim/pass.h>

namespace mim::plug::aie2p {
void register_stages(mim::Flags2Stages&);
}

extern "C" MIM_EXPORT mim::Plugin mim_get_plugin() {
    return {
        "aie2p",
        nullptr, // normalizers
        mim::plug::aie2p::register_stages,
        nullptr // backends
    };
}
