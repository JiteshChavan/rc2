import torch
import pytest
from einops import rearrange

from mamba_ssm.ops.selective_scan_interface import ArceeSelectiveScanFn, mamba_inner_fn, arcee_mamba_inner_ref
import selective_scan_cuda


_RTOL_PROJ = 5e-3
_ATOL_PROJ = 5e-2


@pytest.mark.parametrize('wtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
@pytest.mark.parametrize("is_variable_C", [False, True])
@pytest.mark.parametrize("is_variable_B", [False, True])
def test_mamba_inner_fn(is_variable_B, is_variable_C, seqlen, itype, wtype):
    return_last_state = False
    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # If we haave z, the errors on the weights seem higher
    rtolw = max(rtolw, rtol)
    atolw = max(atolw, atol)

    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    dim = 768 # d_inner
    dstate = 8 # dstate
    dt_rank = 48
    is_complex = wtype == torch.complex64

    # we dont start with any seed for testing vanilla mamba fn
    # h0 = torch.randn(batch_size, dim, dstate, device=device, dtype=wtype, requires_grad=True) # same wtype as A
    xz = torch.randn (batch_size, 2 * dim, seqlen, device=device, dtype=itype, requires_grad=True) # 2 * d_inner  channels (after in_proj)
    conv1d_weight = torch.randn(dim, 1, 3, device=device, dtype=torch.float32, requires_grad=True)
    conv1d_bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    x_proj_weight = torch.randn(dt_rank + (bool(is_variable_B) + bool(is_variable_C)) * dstate * (1 if not is_complex else 2), dim,
                                device=device, dtype=itype, requires_grad=True)
    delta_proj_weight = torch.randn(dim, dt_rank, device=device, dtype=itype, requires_grad=True)
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
    B = (torch.randn(dim, dstate, device=device, dtype=wtype, requires_grad=True) if not is_variable_B else None)
    C = (torch.randn(dim, dstate, device=device, dtype=wtype, requires_grad=True) if not is_variable_C else None)
    D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    delta_bias = (0.5 * torch.randn(dim, device=device, dtype=torch.float32)).requires_grad_()
    B_proj_bias = None
    C_proj_bias = None

    #h0_ref = h0.clone().detach().requires_grad_()
    xz_ref = xz.clone().detach().requires_grad_()
    conv1d_weight_ref = conv1d_weight.clone().detach().requires_grad_()
    conv1d_bias_ref = conv1d_bias.clone().detach().requires_grad_()
    x_proj_weight_ref = x_proj_weight.clone().detach().requires_grad_()
    delta_proj_weight_ref = delta_proj_weight.clone().detach().requires_grad_()

    A_ref = A.clone().detach().requires_grad_()
    B_ref = B.clone().detach().requires_grad_() if not is_variable_B else None
    C_ref = C.clone().detach().requires_grad_() if not is_variable_C else None
    D_ref = D.clone().detach().requires_grad_()
    delta_bias_ref = delta_bias.clone().detach().requires_grad_() if delta_bias is not None else None

    # ref basically calls harness around(wrapper around cuda which is proven to be equivalent to pytorch slow algorith by prior tests)
    out_ref = arcee_mamba_inner_ref (xz_ref, conv1d_weight_ref, conv1d_bias_ref, x_proj_weight_ref, delta_proj_weight_ref,
                                     A_ref, B_ref, C_ref, D_ref,
                                     delta_bias_ref, delta_softplus=True, return_last_state=False, h0=None)
    
    # test function basically calls a different harness around(the exact same cuda which is proven to be equivalent to pytorch slow algorith by prior tests)
    # so in essence we are just testing the harness around the very same cuda kernels here
    # numbers should be much more closer when comparing different harnesses around the exact same kernels than comparing reference pytorch vs cuda kernels
    out_cuda = mamba_inner_fn (xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                                     A, B, C, D,
                                     delta_bias, delta_softplus=True)
    
    if return_last_state:
        out_ref, last_state_ref = out_ref
        out_cuda, last_state_cuda = out_cuda
        assert_and_report_relative_error_at_maxdiff_idx("last_state", last_state_ref, last_state_cuda, rtol=rtol, atol=atol, equal_nan=True)
        print (f"shape of last_state:{last_state_cuda.shape}")
    else:
        last_state_ref = None
        last_state_cuda = None
    
    print (f"out max diff: {(out_ref - out_cuda).abs().max().item()}")
    print (f"out mean diff: {(out_ref - out_cuda).abs().mean().item()}")

    assert_and_report_relative_error_at_maxdiff_idx("inner_out", out_ref, out_cuda, rtol=rtol, atol=atol, equal_nan=True)
    for t in [xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
          A, B, C, D, delta_bias,
          xz_ref, conv1d_weight_ref, conv1d_bias_ref, x_proj_weight_ref, delta_proj_weight_ref,
          A_ref, B_ref, C_ref, D_ref, delta_bias_ref]:
        if t is not None and t.grad is not None:
            t.grad.zero_()
    dout = torch.randn_like(out_cuda)
    if return_last_state:
        dlast_state = torch.randn_like(last_state_cuda)
        torch.autograd.backward((out_ref, last_state_ref), (dout, dlast_state))
        torch.autograd.backward((out_cuda, last_state_cuda), (dout, dlast_state))
    else:
        out_cuda.backward(dout)
        out_ref.backward(dout)

    print(f"dxz max diff : {(xz.grad - xz_ref.grad).abs().max().item()}")

    print(f"dA max diff: {(A.grad - A_ref.grad).abs().max().item()}")
    if not is_variable_B:
        print(f"dB max diff: {(B.grad - B_ref.grad).abs().max().item()}")
        #assert_and_report_relative_error_at_maxdiff_idx("dB", B.grad, B_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    if not is_variable_C:
        print (f"dC max diff: {(C.grad - C_ref.grad).abs().max().item()}")
        #assert_and_report_relative_error_at_maxdiff_idx("dC", C.grad, C_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    print(f"dD max diff: {(D.grad - D_ref.grad).abs().max().item()}")
    print(f"ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}")

    print (f"ddelta_proj_weight max diff: {(delta_proj_weight.grad - delta_proj_weight_ref.grad).abs().max().item()}")
    print(f"dx_proj_weight max diff: {(x_proj_weight.grad - x_proj_weight_ref.grad).abs().max().item()}")
    print (f"dconv1d_weight max diff: {(conv1d_weight.grad - conv1d_weight_ref.grad).abs().max().item()}")
    print (f"dconv1d_bias max diff: {(conv1d_bias.grad - conv1d_bias_ref.grad).abs().max().item()}")
    if return_last_state:
        print (f"h0 seed gradient stats!:")
        print (f"dh0 max diff: {(h0.grad - h0_ref.grad).abs().max().item()}")
        assert_and_report_relative_error_at_maxdiff_idx("dh0", h0.grad, h0_ref.grad, rtol=rtol * 2, atol=atol * 3, equal_nan=True)
    
    
    #assert_and_report_relative_error_at_maxdiff_idx("dxz", xz.grad, xz_ref.grad, rtol=rtol * 2, atol=atol * 3, equal_nan=True)
    assert_and_report_relative_error_at_maxdiff_idx("dA", A.grad, A_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    assert_and_report_relative_error_at_maxdiff_idx("dD", D.grad, D_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    assert_and_report_relative_error_at_maxdiff_idx("ddelta_bias", delta_bias.grad, delta_bias_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    assert_and_report_relative_error_at_maxdiff_idx("ddelta_proj_weight", delta_proj_weight.grad, delta_proj_weight_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    ######## removed from inner fn assert_and_report_relative_error_at_maxdiff_idx("dout_proj_weight", out_proj_weight.grad, out_proj_weight_ref.grad, rtol=_RTOL_PROJ, atol=_ATOL_PROJ, equal_nan=True)
    # assert_and_report_relative_error_at_maxdiff_idx("dx_proj_weight", x_proj_weight.grad, x_proj_weight_ref.grad, rtol=rtol*2, atol=atol*2, equal_nan=True)
    #assert_and_report_relative_error_at_maxdiff_idx("dconv1d_weight", conv1d_weight.grad, conv1d_weight_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    #assert_and_report_relative_error_at_maxdiff_idx("dconv1d_bias", conv1d_bias.grad, conv1d_bias_ref.grad, rtol=rtol*2, atol=atol*3, equal_nan=True)
    print (f"shape out : {out_cuda.shape}")



    
    


def _to64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(torch.complex128 if torch.is_complex(x) else torch.float64)

def assert_and_report_relative_error_at_maxdiff_idx(
    name, ref, test, rtol=1e-3, atol=1e-3,
    eps_f32=1e-12, eps_f16=1e-6, eps_bf16=1e-6, equal_nan=False
):
    assert ref.shape == test.shape, f"{name}: shape mismatch {test.shape} vs {ref.shape}"

    dev = ref.device
    ref = ref.contiguous()
    test = test.contiguous()
    ref64  = _to64(ref)
    test64 = _to64(test)
    diff = (ref64 - test64).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    idx = torch.unravel_index(diff.view(-1).argmax(), diff.shape)

    ref_abs = ref64[idx].abs()
    test_abs = test64[idx].abs()


    if ref.dtype in (torch.float16, torch.complex32) if hasattr(torch, "complex32") else (torch.float16,):
        eps = torch.tensor(eps_f16, device=dev, dtype=torch.float64)
    elif ref.dtype == torch.bfloat16:
        eps = torch.tensor(eps_bf16, device=dev, dtype=torch.float64)
    else:
        eps = torch.tensor(eps_f32, device=dev, dtype=torch.float64)

    denom_ref = torch.clamp(ref_abs, min=eps)
    denom_sym = torch.maximum(torch.maximum(ref_abs, test_abs), eps)

    tol_here = (float(atol) + float(rtol) * ref_abs).item()

    # violation mask across all elements
    per_tol = atol + rtol * ref64.abs()
    violations = (diff > per_tol)
    n_viol = int(violations.sum().item())
    if n_viol:
        # find worst violator by (diff - tol)
        margin = (diff - per_tol)
        worst_idx = torch.unravel_index(margin.view(-1).argmax(), diff.shape)
        worst_ref = ref64[worst_idx].item()
        worst_test = test64[worst_idx].item()
        worst_diff = diff[worst_idx].item()
        worst_tol = per_tol[worst_idx].item()
    else:
        worst_idx = tuple(i.item() for i in idx)
        worst_ref = ref64[idx].item()
        worst_test = test64[idx].item()
        worst_diff = max_diff.item()
        worst_tol = (atol + rtol * ref_abs).item()

    print(
        f"{name}: max_diff={max_diff.item():.6g}, mean_diff={mean_diff.item():.6g}, "
        f"idx_max={tuple(i.item() for i in idx)}, "
        f"rel_at_max(ref)={(max_diff/denom_ref).item():.3e}, rel_at_max(sym)={(max_diff/denom_sym).item():.3e}, "
        f"tol_at_max={tol_here:.3e}; "
        f"violations={n_viol}"
    )
    if n_viol:
        print(
            f"{name}: worst_violation idx={tuple(i.item() for i in worst_idx)}, "
            f"ref={worst_ref:.6g}, test={worst_test:.6g}, "
            f"diff={worst_diff:.6g} > tol={worst_tol:.6g}"
        )

    assert torch.allclose(ref, test, rtol=rtol, atol=atol, equal_nan=equal_nan), f"{name}: {n_viol} element(s) violate tolerance"



if __name__ == "__main__":
    #test_selective_scan(is_variable_B=True, is_variable_C=True, varBC_groups=1, has_D=True, has_z=True, has_delta_bias=True,
    #                    delta_softplus=True, return_last_state=True, seqlen=4096, itype=torch.float32, wtype=torch.float32)
    test_mamba_inner_fn(is_variable_B=False, is_variable_C=False, seqlen=4096, itype=torch.float32, wtype=torch.float32, return_last_state=True)   