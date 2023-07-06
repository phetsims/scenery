/* eslint-disable */
import transform from './shared/transform.js';
import cubic from './shared/cubic.js';
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
${cubic}
${transform}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>s:aX;@group(0)@binding(2)var<storage>b6:array<ak>;struct hq{J:da,S:da,T:da,V:da,ad:h,bJ:j}@group(0)@binding(3)var<storage,read_write>bM:array<hq>;@group(0)@binding(4)var<storage,read_write>eI:array<fv>;var<private>bt:j;fn b0(p:j)->v{return bitcast<v>(vec2(s[bt+p],s[bt+p+f]));}fn b_(p:j)->v{let du=s[bt+p];let x=h(m(du<<16u)>>16u);let y=h(m(du)>>16u);return vec2(x,y);}fn e8(c5:j,p:j)->aL{let am=c5+p*6u;let O=bitcast<L>(vec4(s[am],s[am+f],s[am+2u],s[am+3u]));let bF=bitcast<v>(vec2(s[am+4u],s[am+5u]));return aL(O,bF);}fn fM(x:h)->m{return m(floor(x));}fn fL(x:h)->m{return m(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=s[n.dR+(p>>2u)];bt=n.bt;let b8=(p&3u)*8u;var U=eu(K&((f<<b8)-f));U=bQ(b6[p>>2u],U);var b7=(K>>b8)&255u;let bK=&bM[U.ac];let ad=bitcast<h>(s[n.iB+U.dN]);if(b7&g_)!=e{(*bK).ad=ad;(*bK).bJ=U.bJ;}let aS=b7&ft;if aS!=e{var t:v;var A:v;var H:v;var Q:v;if(b7&g0)!=e{t=b0(U.aI);A=b0(U.aI+2u);if aS>=c4{H=b0(U.aI+4u);if aS==dP{Q=b0(U.aI+6u);}}}else{t=b_(U.aI);A=b_(U.aI+f);if aS>=c4{H=b_(U.aI+2u);if aS==dP{Q=b_(U.aI+3u);}}}let Y=e8(n.c5,U.bJ);t=c0(Y,t);A=c0(Y,A);var l=vec4(min(t,A),max(t,A));if aS==g1{Q=A;H=mix(Q,t,d/3.);A=mix(t,Q,d/3.);}else if aS>=c4{H=c0(Y,H);l=vec4(min(l.xy,H),max(l.zw,H));if aS==dP{Q=c0(Y,Q);l=vec4(min(l.xy,Q),max(l.zw,Q));}else{Q=H;H=mix(A,H,d/3.);A=mix(A,t,d/3.);}}var a2=vec2(c,0.);if ad>=c{a2=.5*ad*vec2(length(Y.O.xz),length(Y.O.yw));l+=vec4(-a2,a2);}let aF=j(ad>=c);eI[N.x]=fv(t,A,H,Q,a2,U.ac,aF);if l.z>l.x||l.w>l.y{atomicMin(&(*bK).J,fM(l.x));atomicMin(&(*bK).S,fM(l.y));atomicMax(&(*bK).T,fL(l.z));atomicMax(&(*bK).V,fL(l.w));}}}`
