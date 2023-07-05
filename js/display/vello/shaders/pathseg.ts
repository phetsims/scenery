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
@group(0)@binding(0)var<uniform>n:aQ;@group(0)@binding(1)var<storage>s:aW;@group(0)@binding(2)var<storage>b7:array<ak>;struct hq{J:db,R:db,S:db,V:db,ad:h,bM:i}@group(0)@binding(3)var<storage,read_write>bP:array<hq>;@group(0)@binding(4)var<storage,read_write>eA:array<fx>;var<private>bu:i;fn b3(p:i)->B{let x=bitcast<h>(s[bu+p]);let y=bitcast<h>(s[bu+p+f]);return vec2(x,y);}fn b2(p:i)->B{let dv=s[bu+p];let x=h(m(dv<<16u)>>16u);let y=h(m(dv)>>16u);return vec2(x,y);}fn fa(c6:i,p:i)->aK{let am=c6+p*6u;let e9=bitcast<h>(s[am]);let e8=bitcast<h>(s[am+f]);let e7=bitcast<h>(s[am+2u]);let e6=bitcast<h>(s[am+3u]);let e5=bitcast<h>(s[am+4u]);let e4=bitcast<h>(s[am+5u]);let K=vec4(e9,e8,e7,e6);let be=vec2(e5,e4);return aK(K,be);}fn fO(x:h)->m{return m(floor(x));}fn fN(x:h)->m{return m(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let L=s[n.dS+(p>>2u)];bu=n.bu;let b9=(p&3u)*8u;var T=em(L&((f<<b9)- f));T=bT(b7[p>>2u],T);var b8=(L>>b9)&255u;let bN=&bP[T.ac];let ad=bitcast<h>(s[n.iB+T.dO]);if(b8&g_)!=e{(*bN).ad=ad;(*bN).bM=T.bM;}let aR=b8&fv;if aR!=e{var t:B;var v:B;var H:B;var P:B;if(b8&g0)!=e{t=b3(T.aH);v=b3(T.aH+2u);if aR>=c5{H=b3(T.aH+4u);if aR==dQ{P=b3(T.aH+6u);}}}else{t=b2(T.aH);v=b2(T.aH+f);if aR>=c5{H=b2(T.aH+2u);if aR==dQ{P=b2(T.aH+3u);}}}let Y=fa(n.c6,T.bM);t=c1(Y,t);v=c1(Y,v);var l=vec4(min(t,v),max(t,v));if aR==g1{P=v;H=mix(P,t,d/3.);v=mix(t,P,d/3.);}else if aR>=c5{H=c1(Y,H);l=vec4(min(l.xy,H),max(l.zw,H));if aR==dQ{P=c1(Y,P);l=vec4(min(l.xy,P),max(l.zw,P));}else{P=H;H=mix(v,H,d/3.);v=mix(v,t,d/3.);}}var a_=vec2(c,0.);if ad>=c{a_=.5*ad*vec2(length(Y.K.xz),length(Y.K.yw));l+=vec4(-a_,a_);}let aE=i(ad>=c);eA[N.x]=fx(t,v,H,P,a_,T.ac,aE);if l.z>l.x||l.w>l.y{atomicMin(&(*bN).J,fO(l.x));atomicMin(&(*bN).R,fO(l.y));atomicMax(&(*bN).S,fN(l.z));atomicMax(&(*bN).V,fN(l.w));}}}`
