/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>u:a0;@group(0)@binding(2)var<storage>aQ:array<al>;@group(0)@binding(3)var<storage,read_write>b8:array<al>;const bZ=8u;const o=256u;var<workgroup>bY:array<al,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let p=N.x;let K=u[n.dS+p];var dW=ew(K);bY[k.x]=dW;for(var i=e;i<bZ;i+=f){workgroupBarrier();if k.x>=f<<i{let au=bY[k.x-(f<<i)];dW=bT(au,dW);}workgroupBarrier();bY[k.x]=dW;}workgroupBarrier();var U=aQ[ap.x];if k.x>e{U=bT(U,bY[k.x-f]);}b8[p]=U;}`