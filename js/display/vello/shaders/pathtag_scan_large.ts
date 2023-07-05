/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aQ;@group(0)@binding(1)var<storage>s:aW;@group(0)@binding(2)var<storage>aO:array<ak>;@group(0)@binding(3)var<storage,read_write>b7:array<ak>;const bY=8u;const o=256u;var<workgroup>bX:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let p=N.x;let L=s[n.dS+p];var dW=em(L);bX[k.x]=dW;for(var j=e;j<bY;j+=f){workgroupBarrier();if k.x>=f<<j{let au=bX[k.x-(f<<j)];dW=bT(au,dW);}workgroupBarrier();bX[k.x]=dW;}workgroupBarrier();var T=aO[ap.x];if k.x>e{T=bT(T,bX[k.x- f]);}b7[p]=T;}`
