/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>j:aM;@group(0)@binding(1)var<storage>n:aS;@group(0)@binding(2)var<storage>aK:array<af>;@group(0)@binding(3)var<storage,read_write>b3:array<af>;const bU=8u;const k=256u;var<workgroup>bk:array<af,k>;var<workgroup>bT:array<af,k>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I,@builtin(local_invocation_id)f:I,@builtin(workgroup_id)al:I){var agg=gZ();if f.x<al.x{agg=aK[f.x];}bk[f.x]=agg;for(var e=0u;e<bU;e+=1u){workgroupBarrier();if f.x+(1u<<e)<k{let ao=bk[f.x+(1u<<e)];agg=bP(agg,ao);}workgroupBarrier();bk[f.x]=agg;}let l=J.x;let H=n[j.dO+l];var dS=ei(H);bT[f.x]=dS;for(var e=0u;e<bU;e+=1u){workgroupBarrier();if f.x>=1u<<e{let ao=bT[f.x-(1u<<e)];dS=bP(ao,dS);}workgroupBarrier();bT[f.x]=dS;}workgroupBarrier();var P=bk[0];if f.x>0u{P=bP(P,bT[f.x- 1u]);}b3[l]=P;}`
