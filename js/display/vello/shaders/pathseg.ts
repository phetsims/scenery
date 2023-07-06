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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>b6:array<ak>;struct hq{J:c9,S:c9,T:c9,V:c9,ad:h,bI:j}@group(0)@binding(3)var<storage,read_write>bL:array<hq>;@group(0)@binding(4)var<storage,read_write>eI:array<fu>;var<private>bt:j;fn b_(p:j)->A{return bitcast<A>(vec2(u[bt+p],u[bt+p+f]));}fn bZ(p:j)->A{let dt=u[bt+p];let x=h(m(dt<<16u)>>16u);let y=h(m(dt)>>16u);return vec2(x,y);}fn fL(x:h)->m{return m(floor(x));}fn fK(x:h)->m{return m(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=u[n.dQ+(p>>2u)];bt=n.bt;let b8=(p&3u)*8u;var U=et(K&((f<<b8)-f));U=bP(b6[p>>2u],U);var b7=(K>>b8)&255u;let bJ=&bL[U.ac];let ad=bitcast<h>(u[n.iB+U.dM]);if(b7&g_)!=e{(*bJ).ad=ad;(*bJ).bI=U.bI;}let aS=b7&fs;if aS!=e{var s:A;var v:A;var H:A;var Q:A;if(b7&g0)!=e{s=b_(U.aH);v=b_(U.aH+2u);if aS>=c4{H=b_(U.aH+4u);if aS==dO{Q=b_(U.aH+6u);}}}else{s=bZ(U.aH);v=bZ(U.aH+f);if aS>=c4{H=bZ(U.aH+2u);if aS==dO{Q=bZ(U.aH+3u);}}}let Y=gI(n.ez,U.bI);s=c0(Y,s);v=c0(Y,v);var l=vec4(min(s,v),max(s,v));if aS==g1{Q=v;H=mix(Q,s,d/3.);v=mix(s,Q,d/3.);}else if aS>=c4{H=c0(Y,H);l=vec4(min(l.xy,H),max(l.zw,H));if aS==dO{Q=c0(Y,Q);l=vec4(min(l.xy,Q),max(l.zw,Q));}else{Q=H;H=mix(v,H,d/3.);v=mix(v,s,d/3.);}}var a2=vec2(c,0.);if ad>=c{a2=.5*ad*vec2(length(Y.P.xz),length(Y.P.yw));l+=vec4(-a2,a2);}let aE=j(ad>=c);eI[N.x]=fu(s,v,H,Q,a2,U.ac,aE);if l.z>l.x||l.w>l.y{atomicMin(&(*bJ).J,fL(l.x));atomicMin(&(*bJ).S,fL(l.y));atomicMax(&(*bJ).T,fK(l.z));atomicMax(&(*bJ).V,fK(l.w));}}}`
