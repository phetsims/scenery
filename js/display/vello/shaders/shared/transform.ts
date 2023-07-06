/* eslint-disable */

export default `struct aP{P:L,b0:B}fn c0(Y:aP,bM:B)->B{return Y.P.xy*bM.x+Y.P.zw*bM.y+Y.b0;}fn fk(Y:aP)->aP{let ir=d/(Y.P.x*Y.P.w-Y.P.y*Y.P.z);let fj=ir*vec4(Y.P.w,-Y.P.y,-Y.P.z,Y.P.x);let iq=mat2x2(fj.xy,fj.zw)*-Y.b0;return aP(fj,iq);}fn dI(a:aP,b:aP)->aP{return aP(a.P.xyxy*b.P.xxzz+a.P.zwzw*b.P.yyww,a.P.xy*b.b0.x+a.P.zw*b.b0.y+a.b0);}fn gH(ey:j,p:j)->aP{let aO=ey+p*6u;let P=bitcast<L>(vec4(u[aO],u[aO+f],u[aO+2u],u[aO+3u]));let b0=bitcast<B>(vec2(u[aO+4u],u[aO+5u]));return aP(P,b0);}`
