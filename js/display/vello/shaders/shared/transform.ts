/* eslint-disable */

export default `struct aL{O:L,bF:v}fn c0(Y:aL,bO:v)->v{return Y.O.xy*bO.x+Y.O.zw*bO.y+Y.bF;}fn fm(Y:aL)->aL{let ir=d/(Y.O.x*Y.O.w-Y.O.y*Y.O.z);let fl=ir*vec4(Y.O.w,-Y.O.y,-Y.O.z,Y.O.x);let iq=mat2x2(fl.xy,fl.zw)*-Y.bF;return aL(fl,iq);}fn dK(a:aL,b:aL)->aL{return aL(a.O.xyxy*b.O.xxzz+a.O.zwzw*b.O.yyww,a.O.xy*b.bF.x+a.O.zw*b.bF.y+a.bF);}`
