/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>u:a0;@group(0)@binding(2)var<storage>aQ:array<al>;@group(0)@binding(3)var<storage,read_write>b7:array<al>;const bY=8u;const o=256u;var<workgroup>bs:array<al,o>;var<workgroup>bX:array<al,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){var agg=ha();if k.x<ap.x{agg=aQ[k.x];}bs[k.x]=agg;for(var i=e;i<bY;i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=bs[k.x+(f<<i)];agg=bS(agg,au);}workgroupBarrier();bs[k.x]=agg;}let p=N.x;let K=u[n.dR+p];var dV=ev(K);bX[k.x]=dV;for(var i=e;i<bY;i+=f){workgroupBarrier();if k.x>=f<<i{let au=bX[k.x-(f<<i)];dV=bS(au,dV);}workgroupBarrier();bX[k.x]=dV;}workgroupBarrier();var U=bs[0];if k.x>e{U=bS(U,bX[k.x-f]);}b7[p]=U;}`
