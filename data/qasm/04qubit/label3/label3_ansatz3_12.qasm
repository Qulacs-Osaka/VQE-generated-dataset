OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.3379086478421582) q[0];
rz(-1.9287523380482967) q[0];
ry(-0.7343341943174099) q[1];
rz(-2.593687789546118) q[1];
ry(1.0393973508006136) q[2];
rz(-1.2165903385084098) q[2];
ry(-2.7600653620049025) q[3];
rz(1.2921617694129142) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.2839959555176055) q[0];
rz(-0.01640499126907002) q[0];
ry(-1.1775628900548194) q[1];
rz(0.624878213863858) q[1];
ry(0.477906789127748) q[2];
rz(-0.10508584052147928) q[2];
ry(2.5436624046122334) q[3];
rz(-0.5581036025998524) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.6027053375262499) q[0];
rz(-0.788648725522612) q[0];
ry(-1.5354087113669488) q[1];
rz(1.427202685924751) q[1];
ry(2.7881545949587903) q[2];
rz(-2.01112196684781) q[2];
ry(2.269869807436842) q[3];
rz(1.4515916342118576) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.9664021857282356) q[0];
rz(-2.7757207040829415) q[0];
ry(-3.051661299494901) q[1];
rz(-1.3523986797503138) q[1];
ry(2.3315768050805543) q[2];
rz(0.6394402108556837) q[2];
ry(2.2801384588195637) q[3];
rz(-0.3476749058274503) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.0420069242724135) q[0];
rz(0.8542644147516669) q[0];
ry(-1.2630811374305577) q[1];
rz(2.9575345973710774) q[1];
ry(1.9783964723966634) q[2];
rz(2.7575735073444005) q[2];
ry(1.1482015992101653) q[3];
rz(0.6478976725577666) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.3154805176513578) q[0];
rz(3.0917895495120913) q[0];
ry(-2.706283611916308) q[1];
rz(-2.5072755803853117) q[1];
ry(0.10032268234449315) q[2];
rz(-2.602405882837743) q[2];
ry(-2.454788905590332) q[3];
rz(1.6119054307123069) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.9259582119464205) q[0];
rz(1.8015538130434512) q[0];
ry(0.2683302609506512) q[1];
rz(-2.1186900942848217) q[1];
ry(-2.5581725439773266) q[2];
rz(-0.12268219510896738) q[2];
ry(-2.2573265407194656) q[3];
rz(0.8550801940674964) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.2510941416372539) q[0];
rz(-1.5267076145879184) q[0];
ry(-1.0148051194543455) q[1];
rz(-1.1322129035187496) q[1];
ry(1.9461759459905201) q[2];
rz(2.4164751024563063) q[2];
ry(2.803920954878916) q[3];
rz(-1.5655674431626034) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.5282422870735388) q[0];
rz(-3.1018783468151563) q[0];
ry(-2.0986483242056533) q[1];
rz(-2.4479380988356043) q[1];
ry(-0.5005784632003031) q[2];
rz(-2.8706848210777527) q[2];
ry(-1.3374266488243265) q[3];
rz(-1.0862409213517026) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.8399437329142279) q[0];
rz(0.8211727442813235) q[0];
ry(0.13389745503192874) q[1];
rz(-2.3748207160733585) q[1];
ry(0.8434305723170896) q[2];
rz(-1.2313553990841726) q[2];
ry(-0.684315367085569) q[3];
rz(1.64304470650619) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.335391815253336) q[0];
rz(0.8622797032409482) q[0];
ry(0.923604088811632) q[1];
rz(-0.9633253617163103) q[1];
ry(-0.1864421586736995) q[2];
rz(-1.2092658710484434) q[2];
ry(2.3537767048002003) q[3];
rz(0.5930079641659676) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.1614367988871983) q[0];
rz(1.9750489575908894) q[0];
ry(2.1974451663742176) q[1];
rz(1.962793346852349) q[1];
ry(1.3609668013006164) q[2];
rz(-2.2781686205244744) q[2];
ry(-0.2580069674641452) q[3];
rz(-1.2472349093547286) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.533677218043438) q[0];
rz(-2.0653812941665306) q[0];
ry(-0.9483530816494756) q[1];
rz(0.5354989929575042) q[1];
ry(1.5284860262875408) q[2];
rz(-1.2014257983943262) q[2];
ry(1.571357739246386) q[3];
rz(0.06303293315823671) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.2424707715492548) q[0];
rz(3.1029061187763505) q[0];
ry(-1.5466694736167714) q[1];
rz(-2.179202396654283) q[1];
ry(-1.0509774732042039) q[2];
rz(-1.8427891748050511) q[2];
ry(-0.6245191233812342) q[3];
rz(0.6001453107776681) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.37612646643172026) q[0];
rz(-1.5524029030207394) q[0];
ry(-1.5545613830080487) q[1];
rz(1.6491158493892666) q[1];
ry(1.916210088935874) q[2];
rz(-2.221112673815669) q[2];
ry(-0.12735205966902619) q[3];
rz(2.0222029989806316) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.8078119395115397) q[0];
rz(0.9171975842563507) q[0];
ry(-0.850911166893836) q[1];
rz(-1.7176116528970296) q[1];
ry(-2.611912434002) q[2];
rz(-0.12548049600794897) q[2];
ry(1.056785589035343) q[3];
rz(-2.3095148447519547) q[3];