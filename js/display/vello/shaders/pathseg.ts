/* eslint-disable */
import pathdata_util from './shared/pathdata_util.js';
import transform from './shared/transform.js';
import cubic from './shared/cubic.js';
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
${cubic}
${transform}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>b6:array<ak>;struct hs{J:c9,S:c9,T:c9,V:c9,ad:h,bJ:j}@group(0)@binding(3)var<storage,read_write>bM:array<hs>;@group(0)@binding(4)var<storage,read_write>eI:array<fu>;var<private>bQ:j;${pathdata_util}
fn fL(x:h)->m{return m(floor(x));}fn fK(x:h)->m{return m(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=u[n.dO+(p>>2u)];bQ=n.bQ;let b8=(p&3u)*8u;var U=et(K&((f<<b8)-f));U=bP(b6[p>>2u],U);var b7=(K>>b8)&255u;let bK=&bM[U.ac];let ad=bitcast<h>(u[n.iE+U.dK]);if(b7&g0)!=e{(*bK).ad=ad;(*bK).bJ=U.bJ;}let aS=b7&fs;if aS!=e{var s:B;var v:B;var H:B;var R:B;if(b7&g1)!=e{s=ck(U.aH);v=ck(U.aH+2u);if aS>=c4{H=ck(U.aH+4u);if aS==dM{R=ck(U.aH+6u);}}}else{s=cj(U.aH);v=cj(U.aH+f);if aS>=c4{H=cj(U.aH+2u);if aS==dM{R=cj(U.aH+3u);}}}let Z=gJ(n.ez,U.bJ);s=c0(Z,s);v=c0(Z,v);var l=vec4(min(s,v),max(s,v));if aS==g2{R=v;H=mix(R,s,d/3.);v=mix(s,R,d/3.);}else if aS>=c4{H=c0(Z,H);l=vec4(min(l.xy,H),max(l.zw,H));if aS==dM{R=c0(Z,R);l=vec4(min(l.xy,R),max(l.zw,R));}else{R=H;H=mix(v,H,d/3.);v=mix(v,s,d/3.);}}var a3=vec2(c,0.);if ad>=c{a3=.5*ad*vec2(length(Z.P.xz),length(Z.P.yw));l+=vec4(-a3,a3);}let aE=j(ad>=c);eI[N.x]=fu(s,v,H,R,a3,U.ac,aE);if l.z>l.x||l.w>l.y{atomicMin(&(*bK).J,fL(l.x));atomicMin(&(*bK).S,fL(l.y));atomicMax(&(*bK).T,fK(l.z));atomicMax(&(*bK).V,fK(l.w));}}}`
