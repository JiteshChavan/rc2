/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
// All strides are in element units of the pointed-to dtype.
#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMScanParamsBase {
    using index_t = uint32_t;

    int batch, seqlen, n_chunks;
    index_t a_batch_stride;
    index_t b_batch_stride;
    index_t out_batch_stride;

    // Common data pointers.
    void *__restrict__ a_ptr;
    void *__restrict__ b_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;
    bool is_variable_B;
    bool is_variable_C;

    bool delta_softplus;

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_z_batch_stride;
    index_t out_z_d_stride;

    // Common data pointers.
    void *__restrict__ A_ptr;
    void *__restrict__ B_ptr;
    void *__restrict__ C_ptr;
    void *__restrict__ D_ptr;
    void *__restrict__ u_ptr;
    void *__restrict__ delta_ptr;
    void *__restrict__ delta_bias_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
    void *__restrict__ z_ptr;
    void *__restrict__ out_z_ptr;

    // If null the kernel falls back to the identity (no seed).
    // shape [batch, dim, dstate]
    // NOTE: If kNRows>1 is ever enabled, the effective “row” index used is (dim_id * kNRows + r).
    // For complex builds, h0_ptr must point to complex_t elements, and strides are in ELEMENTS.
    void *__restrict__ h0_ptr = nullptr;
    index_t h0_batch_stride = 0;
    index_t h0_d_stride = 0;
    index_t h0_dstate_stride = 0;

};

struct SSMParamsBwd: public SSMParamsBase {
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dA_d_stride;
    index_t dA_dstate_stride;
    index_t dB_batch_stride;
    index_t dB_group_stride;
    index_t dB_d_stride;
    index_t dB_dstate_stride;
    index_t dC_batch_stride;
    index_t dC_group_stride;
    index_t dC_d_stride;
    index_t dC_dstate_stride;
    index_t du_batch_stride;
    index_t du_d_stride;
    index_t dz_batch_stride;
    index_t dz_d_stride;
    index_t ddelta_batch_stride;
    index_t ddelta_d_stride;

    // Common data pointers.
    void *__restrict__ dout_ptr;
    void *__restrict__ dA_ptr;
    void *__restrict__ dB_ptr;
    void *__restrict__ dC_ptr;
    void *__restrict__ dD_ptr;
    void *__restrict__ du_ptr;
    void *__restrict__ dz_ptr;
    void *__restrict__ ddelta_ptr;
    void *__restrict__ ddelta_bias_ptr;

    // For complex builds, dh0_ptr must point to complex_t elements, and strides are in ELEMENTS.
    void *__restrict__ dh0_ptr = nullptr;
    index_t dh0_batch_stride = 0;
    index_t dh0_d_stride = 0;
    index_t dh0_dstate_stride = 0;

    // Upstream dL/d(last_state). If null, treated as zeros (no last-state reuse grad).
    // For complex builds, d_last_state_ptr must point to complex_t elements, and strides are in ELEMENTS.
    void *__restrict__ d_last_state_ptr = nullptr;
    index_t d_last_state_batch_stride = 0;
    index_t d_last_state_d_stride = 0;
    index_t d_last_state_dstate_stride = 0;
};
