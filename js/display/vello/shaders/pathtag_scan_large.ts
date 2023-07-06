/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>aN:array<ak>;@group(0)@binding(3)var<storage,read_write>b5:array<ak>;const bU=8u;const o=256u;var<workgroup>bT:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let p=N.x;let K=u[n.dO+p];var dS=er(K);bT[k.x]=dS;for(var i=e;i<bU;i+=f){workgroupBarrier();if k.x>=f<<i{let au=bT[k.x-(f<<i)];dS=bO(au,dS);}workgroupBarrier();bT[k.x]=dS;}workgroupBarrier();var U=aN[ap.x];if k.x>e{U=bO(U,bT[k.x-f]);}b5[p]=U;}`
