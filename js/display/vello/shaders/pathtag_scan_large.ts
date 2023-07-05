/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>s:aX;@group(0)@binding(2)var<storage>aP:array<al>;@group(0)@binding(3)var<storage,read_write>ca:array<al>;const b_=8u;const o=256u;var<workgroup>bZ:array<al,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)at:M){let p=N.x;let L=s[n.dS+p];var dW=em(L);bZ[k.x]=dW;for(var j=e;j<b_;j+=f){workgroupBarrier();if k.x>=f<<j{let aw=bZ[k.x-(f<<j)];dW=bV(aw,dW);}workgroupBarrier();bZ[k.x]=dW;}workgroupBarrier();var U=aP[at.x];if k.x>e{U=bV(U,bZ[k.x-f]);}ca[p]=U;}`
