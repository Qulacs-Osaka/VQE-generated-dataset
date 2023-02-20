OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.6102318836172094) q[0];
rz(-1.8858810153828287) q[0];
ry(-1.6256113458345345) q[1];
rz(-1.8247182043635997) q[1];
ry(3.015714863571693) q[2];
rz(-1.3376998327105825) q[2];
ry(2.2951409555359277) q[3];
rz(3.1046171377120735) q[3];
ry(3.1415926006766552) q[4];
rz(-1.0861315031174301) q[4];
ry(-0.04019930051807329) q[5];
rz(-0.6892161293752751) q[5];
ry(1.06937986670673) q[6];
rz(2.98451345282192) q[6];
ry(-0.8448588985437572) q[7];
rz(-0.017621258449628027) q[7];
ry(3.1415916423179495) q[8];
rz(-2.1700216680286033) q[8];
ry(-0.25981459603402346) q[9];
rz(-1.5878263773282533) q[9];
ry(-2.0111071420384343) q[10];
rz(-0.003406968725420789) q[10];
ry(-0.5940753605628567) q[11];
rz(0.0046439954415354295) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.6502479665711967) q[0];
rz(0.8333362966598817) q[0];
ry(1.9418075510652573) q[1];
rz(1.8146582243785148) q[1];
ry(1.4999918842540143) q[2];
rz(-1.3819254694243988) q[2];
ry(0.6680644505813579) q[3];
rz(0.9849094140026555) q[3];
ry(-1.4854067005551017e-06) q[4];
rz(-0.8884973381510478) q[4];
ry(0.6449309659550799) q[5];
rz(2.035898171128564) q[5];
ry(-3.0610051413383728) q[6];
rz(1.2974704190244999) q[6];
ry(-1.1182127893816238) q[7];
rz(0.6916260169289724) q[7];
ry(-3.141589345086513) q[8];
rz(-2.780202771462864) q[8];
ry(1.5716871397484997) q[9];
rz(0.012401036139697329) q[9];
ry(2.6991439270289543) q[10];
rz(-1.571579062406742) q[10];
ry(-0.245643267989907) q[11];
rz(-1.5613320163661255) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.0333646280615678) q[0];
rz(-0.9977883829826063) q[0];
ry(-2.776220792952501) q[1];
rz(-0.1673484464662467) q[1];
ry(-1.0473816394503481) q[2];
rz(-0.05152236485458757) q[2];
ry(-3.11696689815594) q[3];
rz(2.8561922887091384) q[3];
ry(-3.141592639945198) q[4];
rz(2.1025914782645656) q[4];
ry(0.010344296892092153) q[5];
rz(0.05586480134450867) q[5];
ry(-3.1059573478818) q[6];
rz(2.1127226609163308) q[6];
ry(0.007840837044858537) q[7];
rz(-2.2493491199729374) q[7];
ry(1.5707952361708506) q[8];
rz(1.5708111995711445) q[8];
ry(-3.137327614121556) q[9];
rz(-2.1765257399243083) q[9];
ry(-1.074837982619644) q[10];
rz(-1.5749553914021266) q[10];
ry(-3.06701134038641) q[11];
rz(-2.10300665602266) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.07771402091635073) q[0];
rz(-0.5772426400840199) q[0];
ry(-3.1259870498264744) q[1];
rz(-1.9177624989675162) q[1];
ry(1.5061049406854696) q[2];
rz(3.111847168412306) q[2];
ry(0.029144804928172764) q[3];
rz(1.5023314212909114) q[3];
ry(1.5707966423663036) q[4];
rz(0.0027014353312235784) q[4];
ry(2.5093264028691764) q[5];
rz(0.6286353805700466) q[5];
ry(-9.692237323832842e-07) q[6];
rz(-1.6666623570852659) q[6];
ry(1.5708327174887406) q[7];
rz(0.34191799165292186) q[7];
ry(0.20111386853863422) q[8];
rz(-1.5707721328271234) q[8];
ry(1.5707976652477527) q[9];
rz(-3.1415874422025936) q[9];
ry(1.570796997326779) q[10];
rz(1.5707960227809834) q[10];
ry(3.1415921687294746) q[11];
rz(-0.5418632931820786) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.5909495619790466) q[0];
rz(-2.437425082395974) q[0];
ry(1.5949078916481945) q[1];
rz(-1.4380472402240299) q[1];
ry(1.5707954101364936) q[2];
rz(-1.5707967318184544) q[2];
ry(-2.038663128731732e-07) q[3];
rz(0.3116590716392053) q[3];
ry(0.007276474451301418) q[4];
rz(-1.5734978953805205) q[4];
ry(-0.011332441477728496) q[5];
rz(1.6401087835208845) q[5];
ry(1.0174532860673935e-06) q[6];
rz(-0.013854066025088565) q[6];
ry(8.6914443890862e-06) q[7];
rz(-0.34191763512437745) q[7];
ry(-0.14228746680044857) q[8];
rz(3.1415569556003735) q[8];
ry(-0.16830588558772186) q[9];
rz(3.141586101350172) q[9];
ry(-1.5673501319260246) q[10];
rz(-1.5707965604121594) q[10];
ry(2.706646543913032) q[11];
rz(0.2175456196463718) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.141592450180764) q[0];
rz(3.0206137916432363) q[0];
ry(-3.1415920789658784) q[1];
rz(-2.178698144855301) q[1];
ry(-1.5707967029625163) q[2];
rz(0.8775339611267092) q[2];
ry(3.141592325007919) q[3];
rz(1.610970163403485) q[3];
ry(1.5707948665148748) q[4];
rz(-2.7377623038862304) q[4];
ry(-1.8799958026508364e-06) q[5];
rz(-2.804989511462791) q[5];
ry(-3.141592186117532) q[6];
rz(-0.6300082965721341) q[6];
ry(1.5707615823479042) q[7];
rz(-1.1624224706373993) q[7];
ry(1.5707967334292516) q[8];
rz(0.40378390184455526) q[8];
ry(1.5707960496427784) q[9];
rz(-2.7410953718772593) q[9];
ry(-1.5707959293335994) q[10];
rz(-1.1741132520003101) q[10];
ry(-5.506247361790519e-07) q[11];
rz(-1.385242242592158) q[11];