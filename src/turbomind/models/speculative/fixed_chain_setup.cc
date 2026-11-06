// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/fixed_chain_setup.h"

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/engine/request.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>

namespace turbomind {

FixedChainSetup::FixedChainSetup(const Communicators& comm, int phases, int max_batch_size):
    comm_{comm}, limit_to_accept_len_host_{max_batch_size, kCPUpinned}, data_(phases)
{
    for (FixedChainPhaseData& data : data_) {
        data.draft_extension_q_offsets_host = {max_batch_size + 1, kCPUpinned};
        data.draft_extension_q_offsets      = {max_batch_size + 1, kDEVICE};
        data.draft_extension_k_offsets      = {max_batch_size + 1, kDEVICE};
        data.draft_extension_local_token_nums.assign(comm_.h_dp_group->n_ranks(), 0);
        data.limit_to_accept_len = {max_batch_size, kDEVICE};
    }
}

void FixedChainSetup::Setup(int phase, const Buffer_<Sequence*>& requests, core::BatchCopy& copy)
{
    FixedChainPhaseData& data = data_.at(phase);
    const int batch_size      = requests.size();

    Buffer_<int>& extension = data.draft_extension_q_offsets_host;
    extension[0]            = 0;

    data.refresh_decode  = {};
    data.refresh_prefill = {};

    const int decode_request_count =
        std::find_if(requests.begin(), requests.end(), [](const Sequence* request) {
            return request->submitted->input_len > 1;
        })
        - requests.begin();

    for (int b = 0; b < batch_size; ++b) {
        const SubmittedRow& row = *requests[b]->submitted;

        extension[b + 1] = extension[b] + static_cast<int>(row.is_extension_candidate());
        limit_to_accept_len_host_[b] = row.autoregres;

        auto& partition = b < decode_request_count ? data.refresh_decode : data.refresh_prefill;
        partition.request_count += 1;
        partition.query_count += row.input_len;
        partition.max_query_length = std::max(partition.max_query_length, row.input_len);
        partition.key_capacity_sum += row.key_capacity_end;
        partition.max_key_capacity = std::max(partition.max_key_capacity, row.key_capacity_end);
    }

    data.draft_extension_query_count = extension[batch_size];

    const int attn_dp_rank = comm_.h_dp_group->rank();
    std::fill(data.draft_extension_local_token_nums.begin(), data.draft_extension_local_token_nums.end(), 0);
    data.draft_extension_local_token_nums[attn_dp_rank] = data.draft_extension_query_count;

    if (comm_.h_dp_group->n_ranks() > 1) {
        comm::AllGather(comm_.h_dp_group, data.draft_extension_local_token_nums.data(), 1);
    }

    data.draft_extension_global_query_count =
        std::accumulate(data.draft_extension_local_token_nums.begin(),
                        data.draft_extension_local_token_nums.end(),
                        0);

    data.extension_decode                  = {};
    data.extension_decode.request_count    = batch_size;
    data.extension_decode.query_count      = data.draft_extension_query_count;
    data.extension_decode.max_query_length = data.draft_extension_query_count ? 1 : 0;

    for (int b = 0; b < batch_size; ++b) {
        if (extension[b + 1] == extension[b]) {
            continue;
        }
        const int capacity = requests[b]->submitted->cache_write_end;
        data.extension_decode.key_capacity_sum += capacity;
        data.extension_decode.max_key_capacity = std::max(data.extension_decode.max_key_capacity, capacity);
    }

    copy(extension, batch_size + 1, data.draft_extension_q_offsets);
    copy(limit_to_accept_len_host_, batch_size, data.limit_to_accept_len);

    if (const char* d = std::getenv("TM_SPEC_DUMP"); d && d[0] && d[0] != '0') {
        TM_LOG_WARN("[spec-geom] ph={} bsz={} ext_q={} ext_glob={} rdec(r={},q={}) rpf(r={},q={}) "
                    "edec(r={},q={},maxq={},kcap={},maxk={})",
                    phase,
                    batch_size,
                    data.draft_extension_query_count,
                    data.draft_extension_global_query_count,
                    data.refresh_decode.request_count,
                    data.refresh_decode.query_count,
                    data.refresh_prefill.request_count,
                    data.refresh_prefill.query_count,
                    data.extension_decode.request_count,
                    data.extension_decode.query_count,
                    data.extension_decode.max_query_length,
                    data.extension_decode.key_capacity_sum,
                    data.extension_decode.max_key_capacity);
        for (int b = 0; b < batch_size; ++b) {
            const auto& r   = *requests[b];
            const auto& row = *r.submitted;
            TM_LOG_WARN("[spec-geom]  b={} seq_len={} prompt={} in_len={} qcnt={} kcap_end={} cw_end={} "
                        "vpos={} ext_cand={} autoregres={} ext_off={}",
                        b,
                        r.seq_len,
                        r.prompt_len,
                        row.input_len,
                        row.query_count,
                        row.key_capacity_end,
                        row.cache_write_end,
                        row.verification_positions,
                        (int)row.is_extension_candidate(),
                        (int)row.autoregres,
                        extension[b + 1] - extension[b]);
        }
    }
}

}  // namespace turbomind
