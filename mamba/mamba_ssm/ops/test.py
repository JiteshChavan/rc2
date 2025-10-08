import torch
import pytest
from einops import rearrange

from mamba_ssm.ops.selective_scan_interface import ArceeSelectiveScanFn, arcee_mamba_inner_fn, arcee_mamba_inner_ref, arcee_selective_scan_ref
import selective_scan_cuda


_RTOL_PROJ = 5e-3
_ATOL_PROJ = 5e-2


@pytest.mark.parametrize('wtype', [torch.float32])
@pytest.mark.parametrize('itype', [torch.float32])
@pytest.mark.parametrize('seqlen', [1, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("return_last_state", [True, False])
@pytest.mark.parametrize('has_delta_bias', [True])
@pytest.mark.parametrize('delta_softplus', [True])
@pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize('has_D', [True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("is_variable_C", [True])
@pytest.mark.parametrize("is_variable_B", [True])
@pytest.mark.parametrize("zero_start", [True, False])

def test_selective_scan(is_variable_B, is_variable_C, varBC_groups, has_D, has_z,
                        has_delta_bias, delta_softplus, return_last_state, seqlen, itype, wtype, zero_start):

    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip() # This config is not applicable
    
    device = "cuda"

    if itype == torch.float32:
        rtol, atol = (6e-4, 2e-3)
    elif itype == torch.bfloat16:
        rtol, atol = (3e-2, 5e-2)
    else:
        rtol, atol = (3e-3, 5e-3)
    
    rtolw, atolw = (1e-3, 1e-3)
    if has_z: # if we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    dim = 4
    dstate = 8
    is_complex = wtype == torch.complex64
    
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_() # init and then make it require grad
    A2 = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_() # init and then make it require grad
    A3 = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_() # init and then make it require grad

    if not is_variable_B:
        B_shape = (dim, dstate)
    elif varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen if not is_complex else seqlen*2)
    else:
        B_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype, requires_grad=True)
    B2 = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype, requires_grad=True)
    B3 = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype, requires_grad=True)

    if not is_variable_C:
        C_shape = (dim, dstate)
    elif varBC_groups == 1:
        C_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        C_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    C = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype, requires_grad=True)
    C2 = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype, requires_grad=True)
    C3 = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype, requires_grad=True)

    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
        D2 = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
        D3 = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None
        D2 = None
        D3 = None
    
    if has_z:
        z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
        z2 = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
        z3 = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    else:
        z = None
        z2 = None
        z3 = None
    
    if has_delta_bias:
        delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_() # init, then make it require grad
        delta_bias2 = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_() # init, then make it require grad
        delta_bias3 = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_() # init, then make it require grad
    else:
        delta_bias = None
        delta_bias2 = None
        delta_bias3 = None
    delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_() # init and then make it require grad
    delta2 = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_() # init and then make it require grad
    delta3 = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_() # init and then make it require grad

    # inputs
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    if not zero_start:
        h0 = torch.randn(batch_size, dim, dstate, device=device, dtype=wtype, requires_grad=True) # same type as A
    else:
        h0 = None

    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None

    A2_ref = A2.detach().clone().requires_grad_()
    B2_ref = B2.detach().clone().requires_grad_()
    C2_ref = C2.detach().clone().requires_grad_()
    D2_ref = D2.detach().clone().requires_grad_() if D is not None else None
    z2_ref = z2.detach().clone().requires_grad_() if z is not None else None
    delta2_ref = delta2.detach().clone().requires_grad_()
    delta_bias2_ref = delta_bias2.detach().clone().requires_grad_() if delta_bias is not None else None

    A3_ref = A3.detach().clone().requires_grad_()
    B3_ref = B3.detach().clone().requires_grad_()
    C3_ref = C3.detach().clone().requires_grad_()
    D3_ref = D3.detach().clone().requires_grad_() if D is not None else None
    z3_ref = z3.detach().clone().requires_grad_() if z is not None else None
    delta3_ref = delta3.detach().clone().requires_grad_()
    delta_bias3_ref = delta_bias3.detach().clone().requires_grad_() if delta_bias is not None else None
    
    
    # inputs
    if zero_start:
        h0_ref = None
    else:
        h0_ref = h0.detach().clone().requires_grad_()

    u_ref = u.detach().clone().requires_grad_()

    # weave h0 and u through block 0
    out_ref = arcee_selective_scan_ref (
        u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z=z_ref,
        delta_bias=delta_bias_ref, delta_softplus=delta_softplus, return_last_state=return_last_state, h0=h0_ref
    )
    
    out_cuda = ArceeSelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, h0)

    if return_last_state:
        out_ref, last_state_ref = out_ref
        out, last_state = out_cuda
        last_state_ref = last_state_ref.clone()
        last_state_ref.retain_grad()
        last_state.retain_grad()
    else:
        out_ref = out_ref
        out = out_cuda
        last_state_ref = None
        last_state = None
    out.retain_grad()
    out_ref.retain_grad()
    
    # weave last_state as h0 and out as u for next block
    out2_ref = arcee_selective_scan_ref (
        out_ref, delta2_ref, A2_ref, B2_ref, C2_ref, D2_ref, z=z2_ref,
        delta_bias=delta_bias2_ref, delta_softplus=delta_softplus, return_last_state=return_last_state, h0=last_state_ref
    )
    out2_cuda = ArceeSelectiveScanFn.apply(out, delta2, A2, B2, C2, D2, z2, delta_bias2, delta_softplus, return_last_state, last_state)

    if return_last_state:
        out2_ref, last_state2_ref = out2_ref
        out2, last_state2 = out2_cuda
        last_state2_ref = last_state2_ref.clone()
        last_state2.retain_grad()
        last_state2_ref.retain_grad()
    else:
        out2_ref = out2_ref
        out2 = out2_cuda
        last_state2_ref = None
        last_state2 = None
    out2.retain_grad()
    out2_ref.retain_grad()

    # weave last_state2 as h0 and out2 as u for next block
    out3_ref = arcee_selective_scan_ref(
        out2_ref, delta3_ref, A3_ref, B3_ref, C3_ref, D3_ref, z3_ref, delta_bias3_ref, delta_softplus, return_last_state=return_last_state, h0=last_state2_ref
    )
    out3_cuda = ArceeSelectiveScanFn.apply(out2, delta3, A3, B3, C3, D3, z3, delta_bias3, delta_softplus, return_last_state, last_state2)

    if return_last_state:
        out3_ref, last_state3_ref = out3_ref
        out3, last_state3 = out3_cuda
        last_state3_ref = last_state3_ref.clone()
        last_state3.retain_grad()
        last_state3_ref.retain_grad()
    else:
        out3_ref = out3_ref
        out3 = out3_cuda
        last_state3_ref = None
        last_state3 = None
    out3.retain_grad()
    out3_ref.retain_grad()
    
    assert_and_report_relative_error_at_maxdiff_idx("out", out_ref, out, rtol=rtol, atol=atol)
    assert_and_report_relative_error_at_maxdiff_idx("out2", out2_ref, out2, rtol=rtol, atol=atol)
    assert_and_report_relative_error_at_maxdiff_idx("out3", out3_ref, out3, rtol=rtol, atol=atol)
    print ("Forward activations match closeley within specified tolerances\n")

    if return_last_state:
        assert_and_report_relative_error_at_maxdiff_idx("last_state", last_state_ref, last_state, rtol=rtol, atol=atol)
        assert_and_report_relative_error_at_maxdiff_idx("last_state2", last_state2_ref, last_state2, rtol=rtol, atol=atol)
        assert_and_report_relative_error_at_maxdiff_idx("last_state3", last_state3_ref, last_state3, rtol=rtol, atol=atol)
        print ("Forward last_states match closeley within specified tolerances\n")

    d_out3 = torch.randn_like(out3)
    if return_last_state:
        d_last_state3 = torch.randn_like(last_state3)

    for t in [A_ref,B_ref,C_ref,D_ref,z_ref,delta_ref,delta_bias_ref,h0_ref,u_ref,
          A2_ref,B2_ref,C2_ref,D2_ref,z2_ref,delta2_ref,delta_bias2_ref,
          A3_ref,B3_ref,C3_ref,D3_ref,z3_ref,delta3_ref,delta_bias3_ref, last_state3, last_state3_ref]:
        if t is not None and t.grad is not None: t.grad = None

    
    print ("\nBackwardPass\n:")
    if return_last_state:
        loss_ref = last_state3_ref.mean() + out3_ref.mean()
        loss = last_state3.mean() + out3.mean()
        print("\tbackward through the graph with out and last state gradient tuple")
        torch.autograd.backward((out3, last_state3), (d_out3, d_last_state3))
        torch.autograd.backward((out3_ref, last_state3_ref), (d_out3, d_last_state3))
    else:
        loss_ref = + out3_ref.mean()
        loss = out3.mean()
        out3.backward(d_out3)
        out3_ref.backward(d_out3)
    #loss_ref.backward()
    #loss.backward()
    
    
    print(f"ddelta max diff: {(delta_ref.grad - delta.grad).abs().max().item()}")
    print(f"dA max diff: {(A_ref.grad - A.grad).abs().max().item()}")
    print(f"dB max diff: {(B_ref.grad - B.grad).abs().max().item()}")
    print(f"dC max diff: {(C_ref.grad - C.grad).abs().max().item()}")
    print(f"dout max diff: {(out_ref.grad - out.grad).abs().max().item()}")
    print(f"ddelta2 max diff: {(delta2_ref.grad - delta2.grad).abs().max().item()}")
    print(f"dA2 max diff: {(A2_ref.grad - A2.grad).abs().max().item()}")
    print(f"dB2 max diff: {(B2_ref.grad - B2.grad).abs().max().item()}")
    print(f"dC2 max diff: {(C2_ref.grad - C2.grad).abs().max().item()}")
    print(f"ddelta3 max diff: {(delta3_ref.grad - delta3.grad).abs().max().item()}")
    print(f"dA3 max diff: {(A3_ref.grad - A3.grad).abs().max().item()}")
    print(f"dB3 max diff: {(B3_ref.grad - B3.grad).abs().max().item()}")
    print(f"dC3 max diff: {(C3_ref.grad - C3.grad).abs().max().item()}")
    if has_D:
        print(f"dD max diff: {(D_ref.grad - D.grad).abs().max().item()}")
        print(f"dD2 max diff: {(D2_ref.grad - D2.grad).abs().max().item()}")
        print(f"dD3 max diff: {(D3_ref.grad - D3.grad).abs().max().item()}")
    if has_z:
        print(f"dz max diff: {(z_ref.grad - z.grad).abs().max().item()}")
        print(f"dz2 max diff: {(z2_ref.grad - z2.grad).abs().max().item()}")
        print(f"dz3 max diff: {(z3_ref.grad - z3.grad).abs().max().item()}")
    if has_delta_bias:
        print(f"ddelta_bias max diff: {(delta_bias_ref.grad - delta_bias.grad).abs().max().item()}")
        print(f"ddelta2_bias max diff: {(delta_bias2_ref.grad - delta_bias2.grad).abs().max().item()}")
        print(f"ddelta3_bias max diff: {(delta_bias3_ref.grad - delta_bias3.grad).abs().max().item()}")
    
    print("\nInput Grads:\n")
    if not zero_start:
        print(f"dh0 max diff: {(h0_ref.grad - h0.grad).abs().max().item()}")
    print(f"du max diff: {(u_ref.grad - u.grad).abs().max().item()}")
    
    diffA = (A_ref.grad - A.grad).abs()
    max_idx_flat = diffA.view(-1).argmax()
    idx = torch.unravel_index(max_idx_flat, diffA.shape)  # tuple of indices

    tolA = (atolw * 5) + rtolw * torch.maximum(A_ref.grad.abs(), A.grad.abs())
    print("A diff max:", diffA[idx].item(),
          "| at index:", tuple(i.item() for i in idx))
    print("A tol at that index:", tolA[idx].item())
    print("A violating count:", (diffA > tolA).sum().item())

    assert_and_report_relative_error_at_maxdiff_idx("A.grad", A_ref.grad, A.grad, rtol=rtolw, atol=atolw * 5)
    assert_and_report_relative_error_at_maxdiff_idx("A2.grad", A2_ref.grad, A2.grad, rtol=rtolw, atol=atolw * 5)
    assert_and_report_relative_error_at_maxdiff_idx("A3.grad", A3_ref.grad, A3.grad, rtol=rtolw, atol=atolw * 5)


    assert_and_report_relative_error_at_maxdiff_idx("delta.grad", delta_ref.grad.to(dtype=itype), delta.grad, rtol=rtol * 5, atol=atol * 10)
    assert_and_report_relative_error_at_maxdiff_idx("delta2.grad", delta2_ref.grad.to(dtype=itype), delta2.grad, rtol=rtol * 5, atol=atol * 10)
    assert_and_report_relative_error_at_maxdiff_idx("delta3.grad", delta3_ref.grad.to(dtype=itype), delta3.grad, rtol=rtol * 5, atol=atol * 10)
    assert_and_report_relative_error_at_maxdiff_idx("B.grad", B_ref.grad, B.grad, rtol=rtolw if not is_variable_B else rtol, atol=atolw if not is_variable_B else atol)
    assert_and_report_relative_error_at_maxdiff_idx("B2.grad", B2_ref.grad, B2.grad, rtol=rtolw if not is_variable_B else rtol, atol=atolw if not is_variable_B else atol)
    assert_and_report_relative_error_at_maxdiff_idx("B3.grad", B3_ref.grad, B3.grad, rtol=rtolw if not is_variable_B else rtol, atol=atolw if not is_variable_B else atol)
    
    assert_and_report_relative_error_at_maxdiff_idx("C.grad", C_ref.grad, C.grad, rtol=rtolw if not is_variable_C else rtol, atol=atolw if not is_variable_C else atol)
    assert_and_report_relative_error_at_maxdiff_idx("C2.grad", C2_ref.grad, C2.grad, rtol=rtolw if not is_variable_C else rtol, atol=atolw if not is_variable_C else atol)
    assert_and_report_relative_error_at_maxdiff_idx("C3.grad", C3_ref.grad, C3.grad, rtol=rtolw if not is_variable_C else rtol, atol=atolw if not is_variable_C else atol)
    print ("Gradients on A123,B123, C123 match closely within tolerances\n")

    if has_D:
        assert_and_report_relative_error_at_maxdiff_idx("D.grad", D_ref.grad, D.grad, rtol=rtolw, atol=atolw)
        assert_and_report_relative_error_at_maxdiff_idx("D2.grad", D2_ref.grad, D2.grad, rtol=rtolw, atol=atolw)
        assert_and_report_relative_error_at_maxdiff_idx("D3.grad", D3_ref.grad, D3.grad, rtol=rtolw, atol=atolw)
        print ("Gradients on D123 match closely within tolerances\n")
    if has_z:
        assert_and_report_relative_error_at_maxdiff_idx("z.grad", z_ref.grad, z.grad, rtol=rtolw, atol=atolw)
        assert_and_report_relative_error_at_maxdiff_idx("z2.grad", z2_ref.grad, z2.grad, rtol=rtolw, atol=atolw)
        assert_and_report_relative_error_at_maxdiff_idx("z3.grad", z3_ref.grad, z3.grad, rtol=rtolw, atol=atolw)
        print ("Gradients on z123 match closely within tolerances\n")
    if has_delta_bias:
        assert_and_report_relative_error_at_maxdiff_idx("delta_bias.grad", delta_bias_ref.grad, delta_bias.grad, rtol=rtolw, atol=atolw)
        assert_and_report_relative_error_at_maxdiff_idx("delta_bias2.grad", delta_bias2_ref.grad, delta_bias2.grad, rtol=rtolw, atol=atolw)
        assert_and_report_relative_error_at_maxdiff_idx("delta_bias3.grad", delta_bias3_ref.grad, delta_bias3.grad, rtol=rtolw, atol=atolw)
        print ("Gradients on delta_bias123 match closely within tolerances\n")
    
    if not zero_start:
        print("dh0_ref:", h0_ref.grad[0,0,:10])
        print("dh0_cuda:", h0.grad[0,0,:10])
    # Input grads
    assert_and_report_relative_error_at_maxdiff_idx("u.grad", u_ref.grad.to(dtype=itype), u.grad, rtol=rtol * 2, atol= atol * 2)
    print ("Gradients on inputs u match closely within tolerances\n")
    if not zero_start:
        assert_and_report_relative_error_at_maxdiff_idx("h0.grad", h0_ref.grad, h0.grad, rtol=rtolw, atol=atolw)
        print ("Gradients on inputs h0 match closely within tolerances\n")

    # intermediate grads
    print("testing gradients on intermediates and terminal output tensors:\n")
    assert_and_report_relative_error_at_maxdiff_idx("out3.grad", out3_ref.grad, out3.grad, rtol=rtolw, atol=atolw)
    print("Out3 grads match within tolerances\n")
    
    assert_and_report_relative_error_at_maxdiff_idx("out2.grad", out2_ref.grad, out2.grad, rtol=rtolw, atol=atolw)
    print("Out2 grads match within tolerances\n")
    
    assert_and_report_relative_error_at_maxdiff_idx("out.grad", out_ref.grad, out.grad, rtol=rtolw, atol=atolw)
    print("Out1 grads match within tolerances\n")
    if return_last_state:
        print ("testing gradients on intermediate last_states that we expose for upstream compute:\n")
        assert_and_report_relative_error_at_maxdiff_idx("last_state3.grad", last_state3_ref.grad, last_state3.grad, rtol=rtolw, atol=atolw)
        print("last_state3 grads match within tolerances\n")

        assert_and_report_relative_error_at_maxdiff_idx("last_state2.grad", last_state2_ref.grad, last_state2.grad, rtol=rtolw, atol=atolw)
        print("last_state2 grads match within tolerances\n")

        assert_and_report_relative_error_at_maxdiff_idx("last_state.grad", last_state_ref.grad, last_state.grad, rtol=rtolw, atol=atolw)
        print("last_state grads match within tolerances\n")


@pytest.mark.parametrize('wtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
@pytest.mark.parametrize("is_variable_C", [False, True])
@pytest.mark.parametrize("is_variable_B", [False, True])
@pytest.mark.parametrize("return_last_state", [False, True])
def test_mamba_inner_fn(is_variable_B, is_variable_C, seqlen, itype, wtype, return_last_state):
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

    h0 = torch.randn(batch_size, dim, dstate, device=device, dtype=wtype, requires_grad=True) # same wtype as A
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

    h0_ref = h0.clone().detach().requires_grad_()
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
                                     delta_bias_ref, delta_softplus=True, return_last_state=return_last_state, h0=h0_ref)
    
    # test function basically calls a different harness around(the exact same cuda which is proven to be equivalent to pytorch slow algorith by prior tests)
    # so in essence we are just testing the harness around the very same cuda kernels here
    # numbers should be much more closer when comparing different harnesses around the exact same kernels than comparing reference pytorch vs cuda kernels
    out_cuda = arcee_mamba_inner_fn (xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                                     A, B, C, D,
                                     delta_bias, delta_softplus=True, return_last_state=return_last_state, h0=h0)
    
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
    for t in [h0, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
          A, B, C, D, delta_bias,
          h0_ref, xz_ref, conv1d_weight_ref, conv1d_bias_ref, x_proj_weight_ref, delta_proj_weight_ref,
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
    print (f"h0 seed gradient stats!:")
    print (f"dh0 max diff: {(h0.grad - h0_ref.grad).abs().max().item()}")
    print (f"dh0 cuda mean: {(h0.grad.mean())}, var : {h0.grad.var()}, norm: {h0.grad.norm()}")
    assert_and_report_relative_error_at_maxdiff_idx("dh0", h0.grad, h0_ref.grad, rtol=rtol * 2, atol=atol * 3, equal_nan=True)
    
    
    #assert_and_report_relative_error_at_maxdiff_idx("dxz", xz.grad, xz_ref.grad, rtol=rtol * 2, atol=atol * 3, equal_nan=True)
    assert_and_report_relative_error_at_maxdiff_idx("dA", A.grad, A_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    assert_and_report_relative_error_at_maxdiff_idx("dD", D.grad, D_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    assert_and_report_relative_error_at_maxdiff_idx("ddelta_bias", delta_bias.grad, delta_bias_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    assert_and_report_relative_error_at_maxdiff_idx("ddelta_proj_weight", delta_proj_weight.grad, delta_proj_weight_ref.grad, rtol=rtol, atol=atol, equal_nan=True)
    ######## removed from inner fn assert_and_report_relative_error_at_maxdiff_idx("dout_proj_weight", out_proj_weight.grad, out_proj_weight_ref.grad, rtol=_RTOL_PROJ, atol=_ATOL_PROJ, equal_nan=True)
    #assert_and_report_relative_error_at_maxdiff_idx("dx_proj_weight", x_proj_weight.grad, x_proj_weight_ref.grad, rtol=rtol*2, atol=atol*2, equal_nan=True)
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
    #                    delta_softplus=True, return_last_state=True, seqlen=4096, itype=torch.float32, wtype=torch.float32, zero_start=False)
    test_mamba_inner_fn(is_variable_B=True, is_variable_C=True, seqlen=4096, itype=torch.float32, wtype=torch.float32, return_last_state=True)   