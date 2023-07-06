/* eslint-disable */
import tile from './shared/tile.js';
import drawtag from './shared/drawtag.js';
import bump from './shared/bump.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bump}
${drawtag}
${tile}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>hn:array<L>;@group(0)@binding(3)var<storage,read_write>ao:eE;@group(0)@binding(4)var<storage,read_write>bN:array<cJ>;@group(0)@binding(5)var<storage,read_write>X:array<bu>;const o=256u;var<workgroup>bh:array<j,o>;var<workgroup>jk:j;var<workgroup>fE:j;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fE=atomicLoad(&ao.aq);}let aq=workgroupUniformLoad(&fE);if(aq&eD)!=e{return;}let aC=d/h(cQ);let bt=d/h(bv);let aG=N.x;var bp=cM;if aG<n.cN{bp=u[n.c5+aG];}var J=0;var S=0;var T=0;var V=0;if bp!=cM&&bp!=eu{let l=hn[aG];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bt));T=m(ceil(l.z*aC));V=m(ceil(l.w*bt));}}let fJ=j(clamp(J,0,m(n.aK)));let fI=j(clamp(S,0,m(n.cO)));let fH=j(clamp(T,0,m(n.aK)));let fG=j(clamp(V,0,m(n.cO)));let b9=(fH-fJ)*(fG-fI);var d5=b9;bh[k.x]=b9;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){d5+=bh[k.x-(f<<i)];}workgroupBarrier();bh[k.x]=d5;}if k.x==o-f{let ca=bh[o- f];var a7=atomicAdd(&ao.F,ca);if a7+ca>n.iC{a7=e;atomicOr(&ao.aq,fy);}bN[aG].X=a7;}storageBarrier();let fF=bN[aG|(o- f)].X;storageBarrier();if aG<n.cN{let hp=select(e,bh[k.x- f],k.x>e);let l=vec4(fJ,fI,fH,fG);let W=cJ(l,fF+hp);bN[aG]=W;}let ho=bh[o- f];for(var i=k.x;i<ho;i+=o){X[fF+i]=bu(0,e);}}`
