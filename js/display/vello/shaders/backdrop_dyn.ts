/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>j:aM;@group(0)@binding(1)var<storage>bM:array<cF>;@group(0)@binding(2)var<storage,read_write>S:array<bn>;const k=256u;var<workgroup>gC:array<d,k>;var<workgroup>cW:array<d,k>;var<workgroup>gB:array<d,k>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I,@builtin(local_invocation_id)f:I){let aC=J.x;var ec=0u;if aC<j.cJ{let T=bM[aC];gC[f.x]=T.h.z-T.h.x;ec=T.h.w-T.h.y;gB[f.x]=T.S;}cW[f.x]=ec;for(var e=0u;e<firstTrailingBit(k);e+=1u){workgroupBarrier();if f.x>=(1u<<e){ec+=cW[f.x-(1u<<e)];}workgroupBarrier();cW[f.x]=ec;}workgroupBarrier();let il=cW[k- 1u];for(var cX=f.x;cX<il;cX+=k){var ap=0u;for(var e=0u;e<firstTrailingBit(k);e+=1u){let bB=ap+((k/2u)>>e);if cX>=cW[bB- 1u]{ap=bB;}}let cg=gC[ap];if cg>0u{var eb=cX-select(0u,cW[ap- 1u],ap>0u);var aF=gB[ap]+eb*cg;var a_=S[aF].K;for(var x=1u;x<cg;x+=1u){aF+=1u;a_+=S[aF].K;S[aF].K=a_;}}}}`
