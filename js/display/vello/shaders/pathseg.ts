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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>b4:array<ak>;struct hq{J:c9,S:c9,T:c9,V:c9,ad:h,bH:j}@group(0)@binding(3)var<storage,read_write>bK:array<hq>;@group(0)@binding(4)var<storage,read_write>eH:array<ft>;var<private>bP:j;${pathdata_util}
fn fK(x:h)->m{return m(floor(x));}fn fJ(x:h)->m{return m(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=u[n.dP+(p>>2u)];bP=n.bP;let b6=(p&3u)*8u;var U=es(K&((f<<b6)-f));U=bO(b4[p>>2u],U);var b5=(K>>b6)&255u;let bI=&bK[U.ac];let ad=bitcast<h>(u[n.iB+U.dL]);if(b5&gZ)!=e{(*bI).ad=ad;(*bI).bH=U.bH;}let aS=b5&fr;if aS!=e{var s:B;var v:B;var H:B;var Q:B;if(b5&g_)!=e{s=ck(U.aH);v=ck(U.aH+2u);if aS>=c4{H=ck(U.aH+4u);if aS==dN{Q=ck(U.aH+6u);}}}else{s=cj(U.aH);v=cj(U.aH+f);if aS>=c4{H=cj(U.aH+2u);if aS==dN{Q=cj(U.aH+3u);}}}let Y=gH(n.ey,U.bH);s=c0(Y,s);v=c0(Y,v);var l=vec4(min(s,v),max(s,v));if aS==g0{Q=v;H=mix(Q,s,d/3.);v=mix(s,Q,d/3.);}else if aS>=c4{H=c0(Y,H);l=vec4(min(l.xy,H),max(l.zw,H));if aS==dN{Q=c0(Y,Q);l=vec4(min(l.xy,Q),max(l.zw,Q));}else{Q=H;H=mix(v,H,d/3.);v=mix(v,s,d/3.);}}var a2=vec2(c,0.);if ad>=c{a2=.5*ad*vec2(length(Y.P.xz),length(Y.P.yw));l+=vec4(-a2,a2);}let aE=j(ad>=c);eH[N.x]=ft(s,v,H,Q,a2,U.ac,aE);if l.z>l.x||l.w>l.y{atomicMin(&(*bI).J,fK(l.x));atomicMin(&(*bI).S,fK(l.y));atomicMax(&(*bI).T,fJ(l.z));atomicMax(&(*bI).V,fJ(l.w));}}}`
