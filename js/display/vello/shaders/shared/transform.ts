/* eslint-disable */

export default `struct aP{P:L,b2:A}fn c0(Y:aP,bN:A)->A{return Y.P.xy*bN.x+Y.P.zw*bN.y+Y.b2;}fn fl(Y:aP)->aP{let ir=d/(Y.P.x*Y.P.w-Y.P.y*Y.P.z);let fk=ir*vec4(Y.P.w,-Y.P.y,-Y.P.z,Y.P.x);let iq=mat2x2(fk.xy,fk.zw)*-Y.b2;return aP(fk,iq);}fn dJ(a:aP,b:aP)->aP{return aP(a.P.xyxy*b.P.xxzz+a.P.zwzw*b.P.yyww,a.P.xy*b.b2.x+a.P.zw*b.b2.y+a.b2);}fn gI(ez:j,p:j)->aP{let aO=ez+p*6u;let P=bitcast<L>(vec4(u[aO],u[aO+f],u[aO+2u],u[aO+3u]));let b2=bitcast<A>(vec2(u[aO+4u],u[aO+5u]));return aP(P,b2);}`
