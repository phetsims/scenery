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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>b5:array<ak>;struct hs{J:c9,S:c9,T:c9,V:c9,ad:h,bI:j}@group(0)@binding(3)var<storage,read_write>bL:array<hs>;@group(0)@binding(4)var<storage,read_write>eH:array<fu>;var<private>bP:j;${pathdata_util}
fn fL(x:h)->m{return m(floor(x));}fn fK(x:h)->m{return m(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=u[n.dP+(p>>2u)];bP=n.bP;let b7=(p&3u)*8u;var U=es(K&((f<<b7)-f));U=bO(b5[p>>2u],U);var b6=(K>>b7)&255u;let bJ=&bL[U.ac];let ad=bitcast<h>(u[n.iC+U.dL]);if(b6&g0)!=e{(*bJ).ad=ad;(*bJ).bI=U.bI;}let aS=b6&fs;if aS!=e{var s:B;var v:B;var H:B;var R:B;if(b6&g1)!=e{s=ck(U.aH);v=ck(U.aH+2u);if aS>=c4{H=ck(U.aH+4u);if aS==dN{R=ck(U.aH+6u);}}}else{s=cj(U.aH);v=cj(U.aH+f);if aS>=c4{H=cj(U.aH+2u);if aS==dN{R=cj(U.aH+3u);}}}let Z=gJ(n.ey,U.bI);s=c0(Z,s);v=c0(Z,v);var l=vec4(min(s,v),max(s,v));if aS==g2{R=v;H=mix(R,s,d/3.);v=mix(s,R,d/3.);}else if aS>=c4{H=c0(Z,H);l=vec4(min(l.xy,H),max(l.zw,H));if aS==dN{R=c0(Z,R);l=vec4(min(l.xy,R),max(l.zw,R));}else{R=H;H=mix(v,H,d/3.);v=mix(v,s,d/3.);}}var a2=vec2(c,0.);if ad>=c{a2=.5*ad*vec2(length(Z.P.xz),length(Z.P.yw));l+=vec4(-a2,a2);}let aE=j(ad>=c);eH[N.x]=fu(s,v,H,R,a2,U.ac,aE);if l.z>l.x||l.w>l.y{atomicMin(&(*bJ).J,fL(l.x));atomicMin(&(*bJ).S,fL(l.y));atomicMax(&(*bJ).T,fK(l.z));atomicMax(&(*bJ).V,fK(l.w));}}}`
