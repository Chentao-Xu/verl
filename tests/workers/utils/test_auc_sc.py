import math
import pytest
import torch
from tensordict import TensorDict
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla as ppo
from verl.trainer.ppo.auc_sc_matched import matched_correction, effective_score
from verl.trainer.ppo import auc_sc_adapter as adapter
from verl.workers.utils.auc_sc import combined_loss, head_for_response_logits
from verl.workers.utils import losses
from verl.utils import tensordict_utils as tu

def cfg():
    c = ActorConfig(strategy='fsdp2', ppo_micro_batch_size_per_gpu=1, rollout_n=8,
        policy_loss=PolicyLossConfig(loss_mode='auc_sc'), use_kl_loss=True, kl_loss_coef=.01)
    c.global_batch_info.update(dp_size=1, batch_num_tokens=1, global_batch_size=1)
    return c

@pytest.mark.parametrize('A',[-1.,0.,1.])
@pytest.mark.parametrize('alpha',[0.,.4,1.])
@pytest.mark.parametrize('tail',[False,True])
def test_actual_ppo_autograd_oracle(A,alpha,tail):
    c=cfg(); x=torch.tensor([-.8,.7,-3.,1.2],dtype=torch.float64,requires_grad=True)
    p=x.detach().softmax(-1)
    q=torch.tensor([.4,.03,.5,.07],dtype=x.dtype)
    o=(x.detach()+torch.tensor([1.,-1.,0.,.5])).softmax(-1)
    if tail:
        q=torch.cat((torch.tensor([.6,.3],dtype=x.dtype),.1*p[2:]/p[2:].sum()))
        o=torch.cat((torch.tensor([.03,.07],dtype=x.dtype),.9*p[2:]/p[2:].sum()))
    ids=torch.arange(2 if tail else 4); w=(o/q).clamp(.2,5.).pow(alpha)
    def main(j,weight):
        return ppo(o[j].log().view(1,1),x.log_softmax(-1)[j].view(1,1),
                   torch.tensor([[A]],dtype=x.dtype)*weight,torch.ones(1,1),config=c)[0]
    center=sum(q[j]*torch.autograd.grad(main(j,w[j]),x)[0] for j in range(4))
    ideal=sum(o[j]*torch.autograd.grad(main(j,1.),x)[0] for j in range(4))
    for y in range(4):
        lp=x.log_softmax(-1)
        sc,_,_=matched_correction(trainer_head_logp=lp[ids].view(1,1,-1),
            old_head_logp=o[ids].log().view(1,1,-1),action_log_probs=lp[y].view(1,1),
            advantages=torch.tensor([[A]],dtype=x.dtype),response_mask=torch.ones(1,1),
            topk_ids=ids.view(1,1,-1),topk_log_probs=q[ids].log().view(1,1,-1),
            topk_mask=torch.ones(1,1,len(ids),dtype=torch.bool),loss_agg_mode='token-mean',config=c,
            weight_fn=lambda r:r.clamp(.2,5.).pow(alpha),target='mismatch',return_token_loss=True)
        g=torch.autograd.grad(main(y,w[y])+sc.sum(),x)[0]
        expected=torch.autograd.grad(main(y,w[y]),x)[0]-center+ideal
        torch.testing.assert_close(g,expected,atol=1e-10,rtol=1e-10)

def test_q_equal_old_and_exact_is():
    c=cfg(); lp=torch.tensor([.1,.3,.6],dtype=torch.float64).log().view(1,1,-1).requires_grad_()
    o=torch.tensor([.4,.5,.1],dtype=lp.dtype).view(1,1,-1)
    for q,fn in [(o,lambda r:r.clamp(.2,5.).pow(.4)),(torch.tensor([.6,.2,.2]).view(1,1,-1),lambda r:r)]:
        sc,_,_=matched_correction(trainer_head_logp=lp,old_head_logp=o.log(),action_log_probs=lp[:,:,0],
            advantages=torch.ones(1,1),response_mask=torch.ones(1,1),topk_ids=torch.arange(3).view(1,1,-1),
            topk_log_probs=q.log(),topk_mask=torch.ones(1,1,3,dtype=torch.bool),loss_agg_mode='token-mean',
            config=c,weight_fn=fn,target='mismatch',return_token_loss=True)
        assert sc.abs().max()<1e-7

def test_head_shift_and_grad():
    x=torch.randn(9,160,requires_grad=True)
    ids=torch.arange(128).expand(2,3,128)
    d=TensorDict({'input_ids':torch.nested.nested_tensor_from_jagged(torch.arange(9),torch.tensor([0,4,9])),
        'responses':torch.zeros(2,3,dtype=torch.long),'prompts':torch.zeros(2,3),
        'attention_mask':torch.tensor([[1,1,0,1,1,0],[1,1,0,1,1,1]]),'score_centering_ids':ids},[2])
    h=head_for_response_logits(x,d).values()
    pos=torch.tensor([1,2,5,6,7])
    expected=x[pos].log_softmax(-1)[:,:128]
    torch.testing.assert_close(h[pos],expected)
    torch.testing.assert_close(torch.autograd.grad(h.sum(),x,retain_graph=True)[0],torch.autograd.grad(expected.sum(),x)[0])

def test_engine_loss_microbatch_kl_detach(monkeypatch):
    monkeypatch.setattr(losses,'no_padding_2_padding',lambda x,_:x)
    torch.manual_seed(42); c=cfg(); x=torch.randn(2,3,160,requires_grad=True)
    old=torch.randn(2,3,160).log_softmax(-1).requires_grad_()
    q=torch.randn(2,3,160).log_softmax(-1).requires_grad_()
    ids=q.detach().topk(128,-1).indices
    adv=torch.tensor([[1.,1.,0.],[-1.,-1.,-1.]],requires_grad=True)
    data=TensorDict({'old_log_probs':old[:,:,0], 'rollout_log_probs':q[:,:,0], 'advantages':adv,
        'response_mask':torch.tensor([[1,1,0],[1,1,1]]),'auc_sc_alpha':torch.full((2,3),.4),
        'score_centering_ids':ids,'score_centering_logprobs':q.gather(-1,ids),
        'old_policy_head_log_probs':old.gather(-1,ids),'ref_log_prob':old.detach()[:,:,0]-.1},[2])
    tu.assign_non_tensor(data,dp_size=1,batch_num_tokens=5,global_batch_size=2)
    lp=x.log_softmax(-1); head=lp.gather(-1,ids)
    total,_=losses.ppo_loss(c,{'log_probs':lp[:,:,0],'auc_sc_head':head},data)
    split=sum(losses.ppo_loss(c,{'log_probs':lp[i:i+1,:,0],'auc_sc_head':head[i:i+1]},data[i:i+1])[0] for i in [1,0])
    torch.testing.assert_close(total,split)
    torch.testing.assert_close(torch.autograd.grad(total,x,retain_graph=True)[0],torch.autograd.grad(split,x,retain_graph=True)[0])
    assert torch.autograd.grad(total,(q,old,adv),allow_unused=True)==(None,None,None)

def test_controller_commit_restore(tmp_path):
    b=TensorDict({'old_log_probs':torch.tensor([[-1.,-2.]]),'rollout_log_probs':torch.tensor([[-2.,-1.]]),
        'advantages':torch.ones(1,2),'response_mask':torch.ones(1,2)},[1])
    r=adapter.prepare(b,adapter.ExplorationTISState.create(),1,3)
    assert 'auc_weights' not in b and 'auc_pg_scale' not in b
    assert r.metrics[adapter.PREFIX+'mismatch_trust_pg_scale']==1
    with pytest.raises(RuntimeError):adapter.commit(r,{'actor/optimizer_step_succeeded':[0],'actor/grad_norm':[1]})
    state=adapter.commit(r,{'actor/optimizer_step_succeeded':[1],'actor/grad_norm':[1]})
    adapter.save_state(tmp_path,state,1)
    restored=adapter.load_state(tmp_path,1)
    assert restored==state
    assert adapter.prepare(b,state,2,3).proposed_state==adapter.prepare(b,restored,2,3).proposed_state
